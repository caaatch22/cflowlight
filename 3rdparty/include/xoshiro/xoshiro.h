/// @brief An implementation of the xoshiro/xoroshiro family of pseudorandom
/// number generators.
/// @link  https://nssn.gitlab.io/xoshiro has the complete documentation.
///
/// SPDX-FileCopyrightText:  2023 Nessan Fitzmaurice <nessan.fitzmaurice@me.com>
/// SPDX-License-Identifier: MIT
#pragma once

#include <array>
#include <bit>
#include <chrono>
#include <concepts>
#include <iostream>
#include <iterator>
#include <random>
#include <type_traits>

#ifdef HAVE_BIT_LIB
#include <bit/bit.h>
#endif

namespace xso {

/// @brief A C++ concept that is just enough to distinguish e.g. a
/// std::normal_distribution from a std::vector<..>. The xso::generator's
/// `sample(...)` methods are overloaded on both types of arguments so we need
/// to differentiate!
/// @note  This implementation relies on fact that STL distributions all define
/// a type/tag that STL containers do not!
template <typename D>
concept Distribution = requires { typename D::param_type; };

// --------------------------------------------------------------------------------------------------------------------
// The main xso::generator<StateEngine, OutputFunction> class ...
// --------------------------------------------------------------------------------------------------------------------
/// @brief  This template class marries a state engine with an output function
/// to create a PRNG.
/// @tparam StateEngine Holds the state and, amongst other things, has a @c
/// step() method to advance it.
/// @tparam OutputFunction A functor that reduces the state to a single output
/// word.
/// @note   Most users will not use this class directly but instead reference
/// one of the type aliased PRNG's below.
template <typename StateEngine, typename OutputFunction>
class generator {
 public:
  /// @brief The StateEngine stores the state in a std::array of words of this
  /// specific type.
  using state_type = typename StateEngine::state_type;

  /// @brief The number of words in the underlying state.
  static constexpr std::size_t size() { return StateEngine::size(); }

  /// @brief The number of bits in the underlying state.
  static constexpr std::size_t digits() { return StateEngine::digits(); }

  /// @brief Each call to generator() returns a single unsigned integer of this
  /// type--uint32_t or uint64_t.
  /// @note  Required by the @c UniformRandomBitGenerator concept.
  /// @note  This is always the same as the word type used to store the state.
  using result_type = typename state_type::value_type;

  /// @brief Smallest value this generator can produce.
  /// @note  Required by the @c UniformRandomBitGenerator concept.
  static constexpr result_type min() noexcept { return 0; }

  /// @brief Largest value this generator can produce.
  /// @note  Required by the @c UniformRandomBitGenerator concept.
  static constexpr result_type max() noexcept {
    return std::numeric_limits<result_type>::max();
  }

  /// @brief Default constructor seeds the generator's full state randomly.
  /// @note  This will produce a high quality stream of random outputs that are
  /// different on each run.
  generator() { seed(); }

  /// @brief Construct a generator quickly but *not* well from a single unsigned
  /// integer value.
  /// @note  Seeding from a single value is an easy way to get repeatable random
  /// streams. However, the streams will not be high 'quality' so this should
  /// only be used for prototyping work.
  explicit generator(result_type s) { seed(s); }

  /// @brief Construct and seed from a full state array.
  /// @param s An array of values which shouldn't be all zeros as that is a
  /// fixed point for xoshiro/xoroshiro.
  explicit generator(const state_type &seed_array) { seed(seed_array); }

  /// @brief Seeds the full generator state randomly primarily from
  /// std::random_device.
  /// @note  This will produce a high quality stream of random outputs that are
  /// different on each run.
  void seed() {
    // We will use std::random_device as the principal source of entropy.
    std::random_device dev;

    // Create an array for the seed state
    state_type seed_array;

    // Fill the seed array with calls to dev() ...
    if constexpr (sizeof(result_type) <=
                  sizeof(std::random_device::result_type)) {
      // Each call to dev() will happily rvs_from one of the seed words.
      for (auto &word : seed_array) word = static_cast<result_type>(dev());
    } else {
      // Need more than one call to dev() to rvs_from a seed word--in practice
      // two calls will do the job.
      for (auto &word : seed_array)
        word = static_cast<result_type>(static_cast<uint64_t>(dev()) << 32 |
                                        dev());
    }

    // It is rumored that some std::random_device implementations aren't great.
    // To mitigate that possibility we add data from a call to a high resolution
    // clock for first word.
    using clock_type = std::chrono::high_resolution_clock;
    auto ticks = static_cast<std::uint64_t>(
        clock_type::now().time_since_epoch().count());

    // However, from call to call, ticks only changes in the low order
    // bits--better scramble things a bit!
    ticks = murmur_scramble64(ticks);

    // Fold the scrambled ticks variable into the first seed word.
    seed_array[0] ^= static_cast<result_type>(ticks);

    // Now seed the state ...
    m_engine.seed(seed_array);
  }

  /// @brief Seed this generator quickly but probably *not* well from a single
  /// unsigned integer value.
  /// @note  Seeding from a single value is an easy way to get repeatable random
  /// streams. However, the streams will not be high 'quality' so this should
  /// only be used for prototyping work.
  constexpr void seed(result_type seed) {
    // Create an array for the seed state.
    state_type seed_array;

    // Scramble the seed.
    auto sm64_state = murmur_scramble64(seed);

    // Use SplitMix64 to at least put some values in all the state words.
    for (auto &word : seed_array)
      word = static_cast<result_type>(split_mix64(sm64_state));

    // Use the array to seed the engine
    m_engine.seed(seed_array);
  }

  /// @brief Seed the state from a full array of words.
  /// @param s An array of values which shouldn't be all zeros as that is a
  /// fixed point for xoshiro/xoroshiro.
  constexpr void seed(const state_type &seed_array) {
    m_engine.seed(seed_array);
  }

  /// @brief Advance the state by one step.
  constexpr void step() { m_engine.step(); }

  /// @brief Reduce the current state to get a single @c result_type & then prep
  /// for the next call.
  /// @note  This method is required by the @c UniformRandomBitGenerator
  /// concept.
  constexpr result_type operator()() {
    result_type retval = m_output(m_engine);
    step();
    return retval;
  }

  /// @brief Discard the next @c z iterations in the random number sequence
  /// @note We can do better for large @c z by using one of the `jump` functions
  void discard(std::uint64_t z) {
    for (std::uint64_t i = 0; i < z; ++i) step();
  }

  /// @brief Returns a single integer value from a uniform distribution over
  /// [a,b].
  /// @note  No error checking is done and the behaviour is undefined if a > b.
  template <std::integral T>
  constexpr T sample(T a, T b) {
    return std::uniform_int_distribution<T>{a, b}(*this);
  }

  /// @brief Returns a single real value from a uniform distribution over [a,b).
  /// @note  No error checking is done and the behaviour is undefined if a > b.
  template <std::floating_point T>
  constexpr T sample(T a, T b) {
    return std::uniform_real_distribution<T>{a, b}(*this);
  }

  /// @brief Returns a single index from a uniform distribution over [0, len).
  /// @note  No error checking is done and the behaviour is undefined if len =
  /// 0.
  template <std::integral T>
  constexpr T index(T len) {
    return sample(T{0}, len - 1);
  }

  /// @brief Returns a single value from an iteration--all elements are equally
  /// likely to be returned.
  /// @note  No error checking is done and the behaviour is undefined if e < b.
  template <std::input_iterator T>
  constexpr auto sample(T b, T e) {
    // arithmetic_type of elements in the iteration & handle trivial case.
    auto len = std::distance(b, e);
    if (len < 2) return *b;

    // Pick an index inside the iteration at random & return the corresponding
    // value.
    auto i = index(len);
    std::advance(b, i);
    return *b;
  }

  /// @brief Returns a single value from a container--all elements are equally
  /// likely to be returned.
  template <typename Container>
  constexpr auto sample(const Container &container) {
    return sample(std::cbegin(container), std::cend(container));
  }
  /// @brief Pick n elements from an iteration [b,e) without replacement & put
  /// the chosen samples in dst.
  /// @note  See the docs for @c std::sample(...) for more details.
  /// @note  No error checking is done and the behaviour is undefined if e < b.
  template <std::input_iterator Src, typename Dst>
  constexpr Dst sample(Src b, Src e, Dst dst, std::unsigned_integral auto n) {
    return std::sample(b, e, dst, n, *this);
  }

  /// @brief Pick n elements from a container without replacement & put the
  /// chosen samples in dst.
  /// @note  See the docs for @c std::sample(...) for more details.
  template <typename Src, typename Dst>
  constexpr auto sample(const Src &src, Dst dst,
                        std::unsigned_integral auto n) {
    return sample(std::cbegin(src), std::cend(src), dst, n);
  }

  /// @brief Returns a single random variate drawn from a distribution.
  /// @param dist The distribution in question e.g. a std::normal_distribution
  /// object.
  constexpr auto sample(Distribution auto &dist) { return dist(*this); }

  /// @brief Pushes n samples from a distribution into a destination iterator.
  /// @param dist The distribution in question e.g. a std::normal_distribution
  /// object.
  /// @param dst  An iterator to the start of where we put the n samples.
  template <typename Dst>
  constexpr Dst sample(Distribution auto &dist, Dst dst,
                       std::unsigned_integral auto n) {
    while (n-- != 0) *dst++ = dist(*this);
    return dst;
  }

  /// @brief Roll a dice with an arbitrary arithmetic_type of sides (defaults to
  /// the usual 6).
  constexpr int roll(int n_sides = 6) { return sample(1, n_sides); }

  /// @brief Flip a coin,
  /// @param p Probability of getting a true uniform return--defaults to a fair
  /// 50%.
  bool flip(double p = 0.5) { return std::bernoulli_distribution{p}(*this); }

  /// @brief Shuffles the elements in an iteration.
  /// @note  No error checking is done and the behaviour is undefined if e < b.
  template <std::random_access_iterator Iter>
  constexpr void shuffle(Iter b, Iter e) {
    std::shuffle(b, e, *this);
  }

  /// @brief Shuffles the elements of a container
  template <typename Container>
  constexpr void shuffle(Container &container) {
    return shuffle(std::begin(container), std::end(container));
  }

  /// @brief  Get the characteristic polynomial associated with this generator
  /// (sans its leading coefficient).
  /// @return The precomputed coefficients are returned packed in word form.
  /// @note   The monic characteristic polynomial is c(x) = x^m + p(x) where
  /// degree[p] < m. This returns p(x).
  /// @note   Limited to the precomputed engine variants--to get others use the
  /// `bit` library.
  static constexpr state_type char_poly() { return StateEngine::char_poly(); }

  /// @brief  Get the jump polynomial that will rapidly advance this generator
  /// ahead by J = n or J = 2^n steps.
  /// @param  n We want to jump the generator forward by J = n steps or J = 2^n
  /// steps so J can be huge.
  /// @param  n_is_exponent If true the jump size is 2^n--allows for e.g. 2^100
  /// slots which overflows std::size_t arg.
  /// @return The coefficients of the jump polynomial stored as an array of
  /// words.
  /// @note   Uses the precomputed char-poly c(x) (really uses p(x) where c(x) =
  /// x^m + p(x) and deg[p] < m).
  static constexpr state_type jump_poly(std::size_t n,
                                        bool n_is_exponent = false) {
    // Characteristic poly is monic c(x) = x^m + p(x) and we have precomputed
    // the coefficients of p(x)
    auto p = char_poly();

    // The jump polynomial is x^J mod c(x)--that computation is happy to just be
    // given p(x) instead of c(x).
    return jump_poly(p, n, n_is_exponent);
  }

  /// @brief  Get the jump polynomial that will rapidly advance this generator
  /// ahead by J = n or J = 2^n steps.
  /// @param  p The generator's characteristic polynomial c(x) should be x^m +
  /// p(x) where degree[p] < m.
  /// @param  n We want to jump the generator forward by J = n slots or J = 2^n
  /// slots so J can be huge.
  /// @param  n_is_exponent If true the jump size is 2^n--allows for e.g. 2^100
  /// slots which overflows std::size_t arg.
  /// @return Coefficients of the jump polynomial r(x) = x^n mod c(x) or x^(2^n)
  /// mod c(x) as an array of words.
  static constexpr state_type jump_poly(const state_type &p, std::size_t n,
                                        bool n_is_exponent = false) {
    // Some dimensional information
    using word_type = typename state_type::value_type;
    constexpr std::size_t bits_per_word =
        std::numeric_limits<word_type>::digits;
    constexpr std::size_t n_words = size();
    constexpr std::size_t n_bits = n_words * bits_per_word;

    // Lambda returns the index of the word that holds polynomial coefficient i.
    auto word = [=](std::size_t i) { return i / bits_per_word; };

    // Lambda returns location of polynomial coefficient i inside the word that
    // holds it.
    auto bit = [=](std::size_t i) { return i % bits_per_word; };

    // Lambda returns a mask that isolates polynomial coefficient i within the
    // word that holds it.
    auto mask = [=](std::size_t i) {
      return word_type(word_type{1} << bit(i));
    };

    // Lambda that returns true if polynomial coefficient i is 1.
    auto test = [=](const auto &poly, std::size_t i) -> bool {
      return poly[word(i)] & mask(i);
    };

    // Lambda that sets polynomial coefficient i to 1.
    auto set = [=](auto &poly, std::size_t i) { poly[word(i)] |= mask(i); };

    // Lambda that computes lhs += rhs which in GF(2) is equivalent to  lhs <-
    // lhs^rhs.
    auto add = [=](auto &lhs, const auto &rhs) {
      for (std::size_t i = 0; i < n_words; ++i) lhs[i] ^= rhs[i];
    };

    // Lambda that performs a single place coefficient-shift on the coefficients
    // stored in an array of words. Shift is to the the right if you think the
    // elements are in vector order [v0,v1,v2,v3] -> [0,v0,v1,v2]. Shift is
    // actually to the left when you think in bit order [b3,b2,b1,b0] ->
    // [b2,b1,b0,0].
    auto shift = [=](auto &poly) {
      constexpr std::size_t complement = bits_per_word - 1;
      for (std::size_t i = n_words - 1; i > 0; --i) {
        auto l = static_cast<word_type>(poly[i] << 1);
        auto r = static_cast<word_type>(poly[i - 1] >> complement);
        poly[i] = l | r;
      }
      poly[0] <<= 1;
    };

    // Some work space we use below.
    state_type sum;
    state_type tmp;

    // Lambda that performs an lhs(x) <- (lhs(x)*rhs(x)) mod c(x) step. It make
    // use of the common workspace sum, tmp. Note: a(x)*b(x) mod c(x) = a_0 b(x)
    // | c(x) + a_1 x b(x) | c(x) + ... + a_{m-1} x^{m-1} b(x) | c(x).
    auto multiply_and_mod = [&](auto &lhs, const auto &rhs) {
      sum.fill(0);
      tmp = rhs;
      for (std::size_t i = 0; i < n_bits; ++i) {
        if (test(lhs, i)) add(sum, tmp);
        if (i + 1 < n_bits) {
          bool add_p = test(tmp, n_bits - 1);
          shift(tmp);
          if (add_p) add(tmp, p);
        }
      }
      lhs = sum;
    };

    // Space for the square powers of x each taken mod c(x).
    // Start with s(x) = x | c(x) then s(x) -> x^2 | c(x) then s(x) -> x^4 |
    // c(x) etc.
    state_type s;
    s.fill(0);
    set(s, 1);

    // Case: J = 2^n -- we just do n squaring steps as the jump polynomial is
    // x^(2^n) mod c(x)
    if (n_is_exponent) {
      for (std::size_t j = 0; j < n; ++j) multiply_and_mod(s, s);
      return s;
    }

    // Case J = n: Need to compute r(x) = x^n mod c(x). First create an array
    // for the coefficients of r(x).
    state_type r;
    r.fill(0);

    // Case J = n < n_bits: Then x^n mod c(x) = x^n so we can set the
    // appropriate coefficient and return.
    if (n < n_bits) {
      set(r, n);
      return r;
    }

    // Case J = n = n_bits: Then x^n mod c(x) = p(x).
    if (n == n_bits) return p;

    // Case J = n > n_bits: Need to do repeated squaring starting from the last
    // known spot.
    r = p;
    std::size_t n_left = n - n_bits;

    // And off we go ...
    while (n_left != 0) {
      // Odd n? Do a multiplication step r(x) <- r(x)*s(x) mod c(x) step.
      if ((n_left & 1) == 1) multiply_and_mod(r, s);

      // Are we done?
      n_left >>= 1;
      if (n_left == 0) break;

      // Do a squaring step s(x) <- s(x)^2 mod c(x) step.
      multiply_and_mod(s, s);
    }
    return r;
  }

  /// @brief Rapidly advance this generator by J = n or J = 2^n steps.
  /// @param n We want to jump the generator forward by J = n steps or J = 2^n
  /// steps so J can be huge.
  /// @param n_is_exponent If true the jump size is 2^n--allows for e.g. 2^100
  /// slots which overflows std::size_t arg.
  /// @note  Uses the precomputed characteristic polynomials--for other
  /// generator variants use the `bit` library.
  /// @note  For multiple jumps of size n first precompute the jump polynomial
  /// for n. and then use the next method.
  void jump(std::size_t n, bool n_is_exponent = false) {
    auto r = jump_poly(n, n_is_exponent);
    jump(r);
  }

  /// @brief Jumps the state quickly forward by J steps where J can be huge
  /// (e.g. 2^100)
  /// @param r Coefficients for r(x) = J mod p(x) where c(x) = x^m + p(x) is the
  /// engine's characteristic polynomial.
  /// @note  r is really a bit-vector but is is passed here as a set of words
  /// instead.
  void jump(const state_type &r) {
    // Some work space ...
    state_type sum;
    std::size_t n_words = size();

    // Some help for iterating through the set bits in each word of r.
    std::size_t bits_per_word = std::numeric_limits<result_type>::digits;
    result_type one = 1;

    // Computing the sum r(T).s = r_0.s + r_1.s^1 + ... + r_{m-1} s^{m-1} where
    // s is the current state. T is the engine's transition matrix and we can
    // compute s^{i+1} = T.s^i using the step() method.
    sum.fill(0);
    for (std::size_t i = 0; i < n_words; ++i) {
      // Iterate over the bits in the current r(x) word
      for (std::size_t b = 0; b < bits_per_word; ++b) {
        // If this bit in r is set then we add the current state to the sum.
        if (r[i] & static_cast<result_type>(one << b)) {
          for (std::size_t w = 0; w < n_words; ++w) sum[w] ^= m_engine[w];
        }

        // Compute the next state s^{i+1} by calling the step() method.
        m_engine.step();
      }
    }

    // Store the computed jump back into the state
    m_engine.seed(sum);
  }

  /// @brief Copy the whole state into a preexisting destination.
  constexpr void copy_state(state_type &dst) const { m_engine.copy_state(dst); }

  /// @brief Read-only access to the i'th state word.
  constexpr result_type operator[](std::size_t i) const { return m_engine[i]; }

 private:
  StateEngine m_engine;
  OutputFunction m_output;

  /// @brief  The SplitMix64 random number generator--a simple generator with 64
  /// bits of state.
  /// @param  state The current value of the 64-bit state which is altered by
  /// this function.
  /// @return A 64-bit unsigned random output.
  static constexpr std::uint64_t split_mix64(std::uint64_t &state) {
    std::uint64_t z = (state += 0x9e3779b97f4a7c15);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
    z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
    z = (z ^ (z >> 31));
    return z;
  };

  /// @brief Uses the murmur algorithm to return a word that is a scrambled
  /// version of the 64 input bits.
  static constexpr std::uint64_t murmur_scramble64(std::uint64_t x) {
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdL;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53L;
    x ^= x >> 33;
    return x;
  }

  /// @brief Uses the murmur algorithm to return a word that is a scrambled
  /// version of the 32 input bits.
  static constexpr std::uint64_t murmur_scramble32(std::uint32_t x) {
    x *= 0xcc9e2d51;
    x = (x << 15) | (x >> 17);
    x *= 0x1b873593;
    return x;
  }
};

// --------------------------------------------------------------------------------------------------------------------
// The partition class is used to split one random number stream into a number
// of non-overlapping "partitions".
// --------------------------------------------------------------------------------------------------------------------
/// @brief  Partition a random number stream into a number of non-overlapping
/// sub-streams.
/// @tparam RNG Any generator type that has a state_type, copy constructor and
/// appropriate `jump` methods etc.
template <typename RNG>
class partition {
 public:
  /// @brief The RNG's state is a std::array of this specific type.
  using state_type = typename RNG::state_type;

  /// @brief Get ready to partition the state stream of a generator into
  /// sub-streams.
  /// @param gen The parent generator already seeded.
  /// @param n_partitions The number of non-overlapping partitions we will split
  /// the parent state stream into.
  partition(const RNG &gen, std::size_t n_partitions) : m_gen{gen} {
    // Make sure the requested number of partitions makes sense--silently fix
    // any issues.
    if (n_partitions == 0) n_partitions = 1;

    // How many bits of state in RNG?
    auto n_bits = 8 * sizeof(state_type);

    // The period of the generator is 2^n_bits so each partition ideally has
    // size 2^n_bits / n_partitions. That number will probably overflow a
    // std::size_t so we must keep everything in exponent form. First we find
    // the smallest n such that 2^n >= n_partitions - 1. Note if n_partitions is
    // 128 the following gives n = 7 and does the same if n = 100.
    auto n = static_cast<std::size_t>(std::bit_width(n_partitions - 1));

    // We will actually create 2^n partitions which is probably more than needed
    // but the wastage is negligible. To create those 2^n partitions we must be
    // able to jump ahead 2^(n_bits - n) steps many times.
    auto exponent = n_bits - n;

    // Precompute the jump polynomial to advance the generator 2^exponent steps
    // i.e. along to the next partition.
    m_jump_poly = RNG::jump_poly(exponent, true);
  }

  /// @brief  Get the next sub-stream.
  /// @return A new generator seeded at the start of the next sub-stream of the
  /// parent random number stream.
  RNG next() {
    // We already have a pre-baked generator seeded at the right spot ready to
    // go.
    RNG retval = m_gen;

    // Prep for the next call by jumping the parent copy once more using our
    // precomputed jump-polynomial.
    m_gen.jump(m_jump_poly);

    // And return the pre-baked one ...
    return retval;
  }

 private:
  RNG m_gen;
  state_type m_jump_poly;
};

// --------------------------------------------------------------------------------------------------------------------
// The xoroshiro state engine
// --------------------------------------------------------------------------------------------------------------------
/// @brief  The state engine for the xoroshiro family of pseudorandom generators
/// @tparam N The number of words of state.
/// @tparam T The state is stored as N words of type T where T is some unsigned
/// integer type.
/// @tparam A, B, C These are the parameters used in the `step()` method that
/// advances the state.
template <std::size_t N, std::unsigned_integral T, uint8_t A, uint8_t B,
          uint8_t C>
class xoroshiro {
 public:
  /// @brief The state is stored as a std::array of this type.
  using state_type = std::array<T, N>;

  /// @brief The number of words in the underlying state.
  static constexpr std::size_t size() { return N; }

  /// @brief The number of bits in the underlying state.
  static constexpr std::size_t digits() {
    return N * std::numeric_limits<T>::digits;
  }

  /// @brief Set the state.
  constexpr void seed(const state_type &src) {
    m_state = src;
    m_final = N - 1;
  }

  /// @brief Copy the whole state into a preexisting destination.
  constexpr void copy_state(state_type &dst) const {
    // Need to untangle the ring buffer
    for (std::size_t i = 0; i < N; ++i) dst[i] = operator[](i);
  }

  /// @brief Read-only access to the i'th state word -- will be a ring for
  /// larger N.
  constexpr T operator[](std::size_t i) const {
    return m_state[(i + m_final + 1) % N];
  }

  /// @brief Step the state forward.
  constexpr void step() {
    // Depending on the size of N we either do an explicit or implicit array
    // shuffle of the state array.
    if constexpr (N == 2)
      simple_step();
    else
      clever_step();
  }

  /// @brief  Get the lower-order coefficients of the characteristic polynomial
  /// associated with this state-engine.
  /// @return The precomputed coefficients are returned packed in word form.
  /// @note   The characteristic polynomial is c(x) = x^m + p(x) where degree[p]
  /// < m. This returns p(x).
  /// @note   Limited engine versions were precomputed--to get other variants
  /// use the `bit` library.
  static constexpr state_type char_poly() {
    if constexpr (std::is_same_v<T, uint32_t> && N == 2 && A == 26 && B == 9 &&
                  C == 13) {
      return {0x6e2286c1, 0x53be9da};
    } else if constexpr (std::is_same_v<T, uint64_t> && N == 2 && A == 24 &&
                         B == 16 && C == 37) {
      return {0x95b8f76579aa001, 0x8828e513b43d5};
    } else if constexpr (std::is_same_v<T, uint64_t> && N == 2 && A == 49 &&
                         B == 21 && C == 28) {
      return {0x8dae70779760b081, 0x31bcf2f855d6e5};
    } else if constexpr (std::is_same_v<T, uint64_t> && N == 16 && A == 25 &&
                         B == 27 && C == 36) {
      return {0x5cfeb8cc48ddb211, 0xb73e379d035a06dd,
              0x17d5100a20a0350e, 0x7550223f68f98cac,
              0x29d373b5c5ed3459, 0x3689b412ef70de48,
              0xa1d3b6ee079a7cc6, 0x9bf0b669abd100f8,
              0x955c84e105f60997, 0x6ca140c61889cddd,
              0xabaf68c5fc3a0e4a, 0xa46134526b83adc5,
              0x710704d05683d63,  0x580d080b44b606a2,
              0x8040a0580158a1,   0x800081};
    } else {
      throw std::invalid_argument(
          "xoroshiro characteristic polynomial not pre-computed for given "
          "parameters!");
    }
  }

 private:
  state_type m_state = {
      1};  // The state is an array of words--should never be all zeros!
  std::size_t m_final = N - 1;  // Current location of the final word of state.

  /// @brief Step the state forward using a straight-forward move all the state
  /// words approach
  /// @note  This is an alternative to the clever_step() and is used for small
  /// values of N
  constexpr void simple_step() {
    // Capture the current values in the first and final words of state
    T s0 = m_state[0];
    T s1 = m_state[N - 1];

    // Shift most of the words of state down one slot
    // Note: It could help to unroll this loop at least once for larger N but
    // the shuffle indices method is better
    for (std::size_t i = 0; i < N - 2; ++i) m_state[i] = m_state[i + 1];

    // Update the first and final words of state
    s1 ^= s0;
    m_state[N - 2] = std::rotl(s0, A) ^ (s1 << B) ^ s1;
    m_state[N - 1] = std::rotl(s1, C);
  }

  /// @brief Step the state forward where we shuffle array indices instead of
  /// all the state array words.
  /// @note  This is an alternative to the simple_step() and is used for larger
  /// values of N
  constexpr void clever_step() {
    // Which indices point to the current final & first words of state
    std::size_t i_final = m_final;
    std::size_t i_first = (m_final + 1) % N;

    // Capture the current values in the final & first words of state
    T s_final = m_state[i_final];
    T s_first = m_state[i_first];

    // Update the values for the final & first words of state
    s_final ^= s_first;
    m_state[i_final] = std::rotl(s_first, A) ^ (s_final << B) ^ s_final;
    m_state[i_first] = std::rotl(s_final, C);

    // Step the index of the final word of state--this shuffles the state array
    // down a slot.
    m_final = i_first;
  }
};

// --------------------------------------------------------------------------------------------------------------------
// The xoshiro state engine
// --------------------------------------------------------------------------------------------------------------------
/// @brief The state-engine class for the xoshiro family of pseudorandom
/// generators.
/// @tparam N The number of words of state.
/// @tparam T The state is stored as N words of type T where T is some unsigned
/// integer type.
/// @tparam A, B These are the parameters used in the `step()` method that
/// advances the state.
template <std::size_t N, std::unsigned_integral T, uint8_t A, uint8_t B>
class xoshiro {
 public:
  /// @brief The state is stored as a std::array of this type.
  using state_type = std::array<T, N>;

  /// @brief The number of words in the underlying state.
  static constexpr std::size_t size() { return N; }

  /// @brief The number of bits in the underlying state.
  static constexpr std::size_t digits() {
    return N * std::numeric_limits<T>::digits;
  }

  /// @brief Set the state.
  constexpr void seed(const state_type &src) { m_state = src; }

  /// @brief Copy the whole state into a preexisting destination.
  constexpr void copy_state(state_type &dst) const { dst = m_state; }

  /// @brief Read-only access to the i'th state word.
  constexpr T operator[](std::size_t i) const { return m_state[i]; }

  /// @brief The crucial method that advances the state one step.
  constexpr void step() {
    if constexpr (N == 4) {
      auto tmp = m_state[1] << A;
      m_state[2] ^= m_state[0];
      m_state[3] ^= m_state[1];
      m_state[1] ^= m_state[2];
      m_state[0] ^= m_state[3];
      m_state[2] ^= tmp;
      m_state[3] = std::rotl(m_state[3], B);
    } else if constexpr (N == 8) {
      auto tmp = m_state[1] << A;
      m_state[2] ^= m_state[0];
      m_state[5] ^= m_state[1];
      m_state[1] ^= m_state[2];
      m_state[7] ^= m_state[3];
      m_state[3] ^= m_state[4];
      m_state[4] ^= m_state[5];
      m_state[0] ^= m_state[6];
      m_state[6] ^= m_state[7];
      m_state[6] ^= tmp;
      m_state[7] = std::rotl(m_state[7], B);
    } else {
      // There is no discernible pattern to the way xoshiro works as the number
      // of words of state increases. The step() method for each N has to be
      // hard coded -- this contrasts to xoroshiro engine above. So if we get to
      // here we don't have a formula that works to advance the state and need
      // to fail out. Get a useful'ish error message by pumping a deliberately
      // false condition into static_assert(...).
      static_assert(N < 0,
                    "No xoshiro step() implementation for this number of words "
                    "of state!");
    }
  };

  /// @brief  Get the lower-order coefficients of the characteristic polynomial
  /// associated with this state-engine.
  /// @return The precomputed coefficients are returned packed in word form.
  /// @note   The characteristic polynomial is c(x) = x^m + p(x) where degree[p]
  /// < m. This returns p(x).
  /// @note   Limited engine versions were precomputed--to get others use the
  /// `bit` library.
  static constexpr state_type char_poly() {
    // In practice we have precomputed the jump polynomial for just a few
    // xoshiro's with specific parameters. See the full docs for all the
    // details.
    if constexpr (std::is_same_v<T, uint32_t> && N == 4 && A == 9 && B == 11) {
      return {0xde18fc01, 0x1b489db6, 0x6254b1, 0xfc65a2};
    } else if constexpr (std::is_same_v<T, uint64_t> && N == 4 && A == 17 &&
                         B == 45) {
      return {0x9d116f2bb0f0f001, 0x280002bcefd1a5e, 0x4b4edcf26259f85,
              0x3c03c3f3ecb19};
    } else if constexpr (std::is_same_v<T, uint64_t> && N == 8 && A == 11 &&
                         B == 21) {
      return {0xcf3cff0c00000001, 0x7fdc78d886f00c63, 0xf05e63fca6d7b781,
              0x7a67058e7bbab6f0, 0xf11eef832e32518f, 0x51ba7c47edc758ad,
              0x8f2d27268ce4b20b, 0x500055d8b77f};
    } else {
      throw std::invalid_argument(
          "xoshiro characteristic polynomial not pre-computed for given "
          "parameters!");
    }
  }

 private:
  state_type m_state = {
      1};  // The state as an array of words--should never be all zeros!
};

// --------------------------------------------------------------------------------------------------------------------
// The various output functors that reduce the current state to a single
// unsigned word of output
// --------------------------------------------------------------------------------------------------------------------
/// @brief  The "*" scrambler returns a simple multiple of one of the state
/// words.
/// @tparam S The multiplier.
/// @tparam w Index of the state word in question.
template <auto S, std::size_t w>
struct star {
  /// @brief Reduces the state input to a single word of output.
  constexpr auto operator()(const auto &state) const { return state[w] * S; }
};

/// @brief  The "**" scrambler returns a scrambled version of one of the state
/// words.
/// @tparam S, R, T Parameters in the scramble method.
/// @tparam w Index of the state word being in question.
template <auto S, auto R, auto T, std::size_t w>
struct star_star {
  /// @brief Reduces the state input to a single word of output.
  constexpr auto operator()(const auto &state) const {
    return std::rotl(state[w] * S, R) * T;
  }
};

/// @brief  The "+" scrambler returns the sum of two of the state words.
/// @tparam w0, w1 Indices of the two state words in question.
template <std::size_t w0, std::size_t w1>
struct plus {
  /// @brief Reduces the state input to a single word of output.
  constexpr auto operator()(const auto &state) const {
    return state[w0] + state[w1];
  }
};

/// @brief  The "++" scrambler returns a scrambled version of two of the state
/// words.
/// @tparam R Parameter in the scramble method.
/// @tparam w0, w1 Indices of the two state words in question.
template <auto R, std::size_t w0, std::size_t w1>
struct plus_plus {
  /// @brief Reduces the state input to a single word of output.
  constexpr auto operator()(const auto &state) const {
    return std::rotl(state[w0] + state[w1], R) + state[w0];
  }
};

// --------------------------------------------------------------------------------------------------------------------
// Type aliases for the 17 generators talked about in the Black & Vigna paper
// --------------------------------------------------------------------------------------------------------------------
// clang-format off
// First we define the preferred versions of the xoshiro engines with specific choices for A & B.
using xoshiro_4x32              = xoshiro<4, uint32_t, 9,  11>;
using xoshiro_4x64              = xoshiro<4, uint64_t, 17, 45>;
using xoshiro_8x64              = xoshiro<8, uint64_t, 11, 21>;

// And the preferred versions of the xoshiro engines with specific choices for A, B & C.
using xoroshiro_2x32            = xoroshiro<2,  uint32_t, 26, 9,  13>;
using xoroshiro_2x64            = xoroshiro<2,  uint64_t, 24, 16, 37>;
using xoroshiro_2x64b           = xoroshiro<2,  uint64_t, 49, 21, 28>;  // Alternative for 2x64 case
using xoroshiro_16x64           = xoroshiro<16, uint64_t, 25, 27, 36>;

// Preferred/analyzed versions of the xoshiro PRNG's
using xoshiro_4x32_plus         = generator<xoshiro_4x32, plus<0, 3>>;
using xoshiro_4x32_plus_plus    = generator<xoshiro_4x32, plus_plus<7, 0, 3>>;
using xoshiro_4x32_star_star    = generator<xoshiro_4x32, star_star<5, 7, 9, 1>>;
using xoshiro_4x64_plus         = generator<xoshiro_4x64, plus<0, 3>>;
using xoshiro_4x64_plus_plus    = generator<xoshiro_4x64, plus_plus<23, 0, 3>>;
using xoshiro_4x64_star_star    = generator<xoshiro_4x64, star_star<5, 7, 9, 1>>;
using xoshiro_8x64_plus         = generator<xoshiro_8x64, plus<2, 0>>;
using xoshiro_8x64_plus_plus    = generator<xoshiro_8x64, plus_plus<17, 2, 0>>;
using xoshiro_8x64_star_star    = generator<xoshiro_8x64, star_star<5, 7, 9, 1>>;

// Preferred/analyzed versions of the xoroshiro PRNG's
using xoroshiro_2x32_star       = generator<xoroshiro_2x32,  star<0x9E3779BB, 0>>;
using xoroshiro_2x32_star_star  = generator<xoroshiro_2x32,  star_star<0x9E3779BBu, 5, 5, 0>>;
using xoroshiro_2x64_plus       = generator<xoroshiro_2x64,  plus<0, 1>>;
using xoroshiro_2x64_plus_plus  = generator<xoroshiro_2x64b, plus_plus<17, 0, 1>>;
using xoroshiro_2x64_star_star  = generator<xoroshiro_2x64,  star_star<5, 7, 9, 0>>;
using xoroshiro_16x64_plus_plus = generator<xoroshiro_16x64, plus_plus<23, 15, 0>>;
using xoroshiro_16x64_star      = generator<xoroshiro_16x64, star<0x9e3779b97f4a7c13, 0>>;
using xoroshiro_16x64_star_star = generator<xoroshiro_16x64, star_star<5, 7, 9, 0>>;
// clang-format on

/// @brief The "default" 32-bit output generator--used as @c xso::rng32
using rng32 = xoshiro_4x32_star_star;

/// @brief The "default" 64-bit output generator--used as @c xso::rng64
using rng64 = xoshiro_4x64_star_star;

/// @brief The "overall default" generator--used as @c xso::rng
using rng = rng64;

// --------------------------------------------------------------------------------------------------------------------
// Non-member functions we can define if the `bit` library is available ...
// --------------------------------------------------------------------------------------------------------------------
#ifdef HAVE_BIT_LIB

/// @brief Returns the transition matrix for a StateEngine/RNG type as a
/// bit::matrix.
template <typename StateEngine>
bit::matrix<> transition_matrix() {
  // The transition matrix will be a square n_bits x n_bits matrix over GF(2).
  using state_type = typename StateEngine::state_type;
  std::size_t n_bits = 8 * sizeof(state_type);
  bit::matrix retval(n_bits, n_bits);

  // Some work-space in word and bit space
  state_type words;
  bit::vector bits(n_bits);

  // Create an instance of the StateEngine.
  StateEngine engine;

  // We get the columns of the transition matrix by looking  at the action of
  // step() on all the unit states.
  for (std::size_t k = 0; k < n_bits; ++k) {
    // Create the k'th unit state (i.e. the state just has the k'th bit set and
    // all others are zero)
    bits.reset();
    bits.set(k);

    // Seed the engine from that k'th unit state--first translating the bits to
    // words
    bit::copy(bits, words);
    engine.seed(words);

    // Advance that k'th unit state one step
    engine.step();

    // Grab the resulting state and convert it back to a bit-vector
    engine.copy_state(words);
    bit::copy(words, bits);

    // Store those bits into column k of the transition matrix.
    // Note that columnar access for a bit::matrix must be done element by
    // element.
    for (std::size_t i = 0; i < n_bits; ++i) retval(i, k) = bits[i];
  }
  return retval;
}

/// @brief Returns the coefficients of a characteristic polynomial for a
/// StateEngine/RNG as a bit::vector.
/// @note  If the transition matrix is m X m then the return will have size
/// m+1--leading coefficient should be 1.
template <typename StateEngine>
bit::vector<> char_poly() {
  auto c = transition_matrix<StateEngine>();
  return bit::characteristic_polynomial(c);
}

/// @brief  Returns a jump polynomial that moves a StateEngine/RNG type J steps
/// ahead in its random number stream.
/// @param  c The precomputed coefficients for the characteristic polynomial of
/// this StateEngine/RNG type.
/// @param  n We want to jump by J = n steps (n may be very large) or J = 2^n
/// steps (for really huge jumps).
/// @param  n_is_exponent If true we want to jump by 2^n steps--allows for say J
/// = 2^100 which overflows normal ints.
/// @return Returns the coefficients of the jump polynomial r(x) = x^J mod c(x).
template <typename StateEngine>
bit::vector<> jump_poly(const bit::vector<> &c, std::size_t n,
                        bool n_is_exponent = false) {
  return bit::polynomial_mod(n, c, n_is_exponent);
}

/// @brief  Returns a jump polynomial that moves StateEngine type J steps ahead
/// in its random number stream.
/// @param  n We want to jump by J = n steps (n may be very large) or J = 2^n
/// steps (for really huge jumps).
/// @param  n_is_exponent If true we want to jump by 2^n steps--allows for say J
/// = 2^100 which overflows normal ints.
/// @note   For multiple n's use the version that takes a precomputed
/// characteristic polynomial.
/// @return Returns coefficients of jump polynomial r(x) = x^J mod c(x) where
/// c(x) is the characteristic polynomial.
template <typename StateEngine>
bit::vector<> jump_poly(std::size_t n, bool n_is_exponent = false) {
  auto c = char_poly<StateEngine>();
  return jump_poly<StateEngine>(c, n, n_is_exponent);
}

/// @brief Jumps a state-engine/RNG ahead in its random number stream by J
/// steps.
/// @param r Precomputed coefficients for x^J mod c(x) where c(x) is the
/// generators's characteristic polynomial.
template <typename StateEngine>
void jump(StateEngine &engine, const bit::vector<> &r) {
  // Some work space.
  using state_type = typename StateEngine::state_type;
  state_type state;
  state_type sum;

  // Computing the sum [r_0 + r_1 T + ... + r_{m-1} T^{m-1}].s where s is the
  // current state. T is the engine's transition matrix and we can compute
  // s^{i+1} = T.s^i using the step() method.
  if constexpr (std::is_integral_v<state_type>) {
    sum = 0;
    for (std::size_t i = 0; i < r.size(); ++i) {
      if (r[i]) {
        engine.copy_state(state);
        sum ^= state;
      }
      engine.step();
    }
  } else {
    sum.fill(0);
    for (std::size_t i = 0; i < r.size(); ++i) {
      if (r[i])
        for (std::size_t w = 0; w < StateEngine::size(); ++w)
          sum[w] ^= engine[w];
      engine.step();
    }
  }

  // Perform the computed jump by reseeding the engine from the computed sum ...
  engine.seed(sum);
}

/// @brief Jumps a state-engine/RNG ahead in its random number stream by J
/// steps.
/// @param n We want to jump by J = n steps (n may be very large) or J = 2^n
/// steps (for really huge jumps).
/// @param n_is_exponent If true we want to jump by 2^n steps--allows for say N
/// = 2^100 which overflows normal ints.
/// @note  For multiple jumps use the version that is passed a precomputed jump
/// polynomial.
template <typename StateEngine>
void jump(StateEngine &engine, std::size_t n, bool n_is_exponent = false) {
  auto r = jump_poly<StateEngine>(n, n_is_exponent);
  jump<StateEngine>(engine, r);
}

#endif  // HAVE_BIT_LIB

}  // namespace xso