package jpountz;

import java.util.concurrent.TimeUnit;

import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.Fork;
import org.openjdk.jmh.annotations.Level;
import org.openjdk.jmh.annotations.Measurement;
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.Warmup;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.ShortVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

@OutputTimeUnit(TimeUnit.MICROSECONDS)
@Warmup(iterations = 5, time = 1, timeUnit = TimeUnit.SECONDS)
@Measurement(iterations = 5, time = 1, timeUnit = TimeUnit.SECONDS)
@Fork(value = 1, jvmArgsPrepend = {"--add-modules=jdk.incubator.vector"})
@State(Scope.Benchmark)
public class DotProductBenchmark {

  @Setup(Level.Trial)
  public void setup() {
    sanity();
  }

  public void sanity() {
    for (int size : new int[] { 384, 1024}) {
      var state = new SimilarityState();
      state.size = size;
      state.setup();
      float expectedValue = scalarFloat(state);
      assertEqual(expectedValue, scalarBFloat16(state));
      assertEqual(expectedValue, luceneScalarFloat(state));
      assertEqual(expectedValue, luceneScalarBFloat16(state));
      assertEqual(expectedValue, vectorizedFloat(state));
      assertEqual(expectedValue, vectorizedBFloat16Emulation(state));
      assertEqual(expectedValue, luceneVectorizedFloat(state));
    }
  }

  static void assertEqual(float expectedValue, float actualValue) {
    float relativeDelta = Math.abs(expectedValue - actualValue) / expectedValue;
    if (relativeDelta < 0.01 == false) {
      throw new AssertionError("Expected: " + expectedValue + ", got" + actualValue);
    }
  }

  private static float decodeBFloat16(short bits) {
    return Float.intBitsToFloat(Short.toUnsignedInt(bits) << 16);
  }

  /**
   * Naive scalar dot product.
   */
  @Benchmark
  public float scalarFloat(SimilarityState state) {
    float[] a = state.a;
    float[] b = state.b;
    float acc = 0;
    for (int i = 0; i < a.length; ++i) {
      acc += a[i] * b[i];
    }
    return acc;
  }

  /**
   * Naiv scalar dot product on bfloat16.
   */
  @Benchmark
  public float scalarBFloat16(SimilarityState state) {
    short[] a = state.a16;
    short[] b = state.b16;
    float acc = 0;
    for (int i = 0; i < a.length; ++i) {
      acc += decodeBFloat16(a[i]) * decodeBFloat16(b[i]);
    }
    return acc;
  }

  /**
   * Naive vectorized dot product.
   */
  @Benchmark
  public float vectorizedFloat(SimilarityState state) {
    float[] a = state.a;
    float[] b = state.b;
    FloatVector acc = FloatVector.zero(FloatVector.SPECIES_PREFERRED);
    int upTo = FloatVector.SPECIES_PREFERRED.loopBound(a.length);
    int i;
    for (i = 0; i < upTo; i += FloatVector.SPECIES_PREFERRED.length()) {
      FloatVector aNext = FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, a, i);
      FloatVector bNext = FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, b, i);
      acc = acc.add(aNext.mul(bNext));
    }
    float res = acc.reduceLanes(VectorOperators.ADD);
    for (; i < a.length; ++i) {
      res += a[i] * b[i];
    }
    return res;
  }

  /**
   * Lucene's implementation of a scalar dot product.
   */
  @Benchmark
  public float luceneScalarFloat(SimilarityState state) {
    float[] a = state.a;
    float[] b = state.b;
    float res = 0f;
    /*
     * If length of vector is larger than 8, we use unrolled dot product to accelerate the
     * calculation.
     */
    int i;
    for (i = 0; i < a.length % 8; i++) {
      res += b[i] * a[i];
    }
    if (a.length < 8) {
      return res;
    }
    for (; i + 31 < a.length; i += 32) {
      res +=
          b[i + 0] * a[i + 0]
              + b[i + 1] * a[i + 1]
                  + b[i + 2] * a[i + 2]
                      + b[i + 3] * a[i + 3]
                          + b[i + 4] * a[i + 4]
                              + b[i + 5] * a[i + 5]
                                  + b[i + 6] * a[i + 6]
                                      + b[i + 7] * a[i + 7];
      res +=
          b[i + 8] * a[i + 8]
              + b[i + 9] * a[i + 9]
                  + b[i + 10] * a[i + 10]
                      + b[i + 11] * a[i + 11]
                          + b[i + 12] * a[i + 12]
                              + b[i + 13] * a[i + 13]
                                  + b[i + 14] * a[i + 14]
                                      + b[i + 15] * a[i + 15];
      res +=
          b[i + 16] * a[i + 16]
              + b[i + 17] * a[i + 17]
                  + b[i + 18] * a[i + 18]
                      + b[i + 19] * a[i + 19]
                          + b[i + 20] * a[i + 20]
                              + b[i + 21] * a[i + 21]
                                  + b[i + 22] * a[i + 22]
                                      + b[i + 23] * a[i + 23];
      res +=
          b[i + 24] * a[i + 24]
              + b[i + 25] * a[i + 25]
                  + b[i + 26] * a[i + 26]
                      + b[i + 27] * a[i + 27]
                          + b[i + 28] * a[i + 28]
                              + b[i + 29] * a[i + 29]
                                  + b[i + 30] * a[i + 30]
                                      + b[i + 31] * a[i + 31];
    }
    for (; i + 7 < a.length; i += 8) {
      res +=
          b[i + 0] * a[i + 0]
              + b[i + 1] * a[i + 1]
                  + b[i + 2] * a[i + 2]
                      + b[i + 3] * a[i + 3]
                          + b[i + 4] * a[i + 4]
                              + b[i + 5] * a[i + 5]
                                  + b[i + 6] * a[i + 6]
                                      + b[i + 7] * a[i + 7];
    }
    return res;
  }

  /**
   * Adaptation of Lucene's implementation of a scalar dot product, but with bfloat16 instead of float.
   */
  @Benchmark
  public float luceneScalarBFloat16(SimilarityState state) {
    short[] a = state.a16;
    short[] b = state.b16;
    float res = 0f;
    /*
     * If length of vector is larger than 8, we use unrolled dot product to accelerate the
     * calculation.
     */
    int i;
    for (i = 0; i < a.length % 8; i++) {
      res += decodeBFloat16(b[i]) * decodeBFloat16(a[i]);
    }
    if (a.length < 8) {
      return res;
    }
    for (; i + 31 < a.length; i += 32) {
      res +=
          decodeBFloat16(b[i + 0]) * decodeBFloat16(a[i + 0])
          + decodeBFloat16(b[i + 1]) * decodeBFloat16(a[i + 1])
          + decodeBFloat16(b[i + 2]) * decodeBFloat16(a[i + 2])
          + decodeBFloat16(b[i + 3]) * decodeBFloat16(a[i + 3])
          + decodeBFloat16(b[i + 4]) * decodeBFloat16(a[i + 4])
          + decodeBFloat16(b[i + 5]) * decodeBFloat16(a[i + 5])
          + decodeBFloat16(b[i + 6]) * decodeBFloat16(a[i + 6])
          + decodeBFloat16(b[i + 7]) * decodeBFloat16(a[i + 7]);
      res +=
          decodeBFloat16(b[i + 8]) * decodeBFloat16(a[i + 8])
          + decodeBFloat16(b[i + 9]) * decodeBFloat16(a[i + 9])
          + decodeBFloat16(b[i + 10]) * decodeBFloat16(a[i + 10])
          + decodeBFloat16(b[i + 11]) * decodeBFloat16(a[i + 11])
          + decodeBFloat16(b[i + 12]) * decodeBFloat16(a[i + 12])
          + decodeBFloat16(b[i + 13]) * decodeBFloat16(a[i + 13])
          + decodeBFloat16(b[i + 14]) * decodeBFloat16(a[i + 14])
          + decodeBFloat16(b[i + 15]) * decodeBFloat16(a[i + 15]);
      res +=
          decodeBFloat16(b[i + 16]) * decodeBFloat16(a[i + 16])
          + decodeBFloat16(b[i + 17]) * decodeBFloat16(a[i + 17])
          + decodeBFloat16(b[i + 18]) * decodeBFloat16(a[i + 18])
          + decodeBFloat16(b[i + 19]) * decodeBFloat16(a[i + 19])
          + decodeBFloat16(b[i + 20]) * decodeBFloat16(a[i + 20])
          + decodeBFloat16(b[i + 21]) * decodeBFloat16(a[i + 21])
          + decodeBFloat16(b[i + 22]) * decodeBFloat16(a[i + 22])
          + decodeBFloat16(b[i + 23]) * decodeBFloat16(a[i + 23]);
      res +=
          decodeBFloat16(b[i + 24]) * decodeBFloat16(a[i + 24])
          + decodeBFloat16(b[i + 25]) * decodeBFloat16(a[i + 25])
          + decodeBFloat16(b[i + 26]) * decodeBFloat16(a[i + 26])
          + decodeBFloat16(b[i + 27]) * decodeBFloat16(a[i + 27])
          + decodeBFloat16(b[i + 28]) * decodeBFloat16(a[i + 28])
          + decodeBFloat16(b[i + 29]) * decodeBFloat16(a[i + 29])
          + decodeBFloat16(b[i + 30]) * decodeBFloat16(a[i + 30])
          + decodeBFloat16(b[i + 31]) * decodeBFloat16(a[i + 31]);
    }
    for (; i + 7 < a.length; i += 8) {
      res +=
          decodeBFloat16(b[i + 0]) * decodeBFloat16(a[i + 0])
          + decodeBFloat16(b[i + 1]) * decodeBFloat16(a[i + 1])
          + decodeBFloat16(b[i + 2]) * decodeBFloat16(a[i + 2])
          + decodeBFloat16(b[i + 3]) * decodeBFloat16(a[i + 3])
          + decodeBFloat16(b[i + 4]) * decodeBFloat16(a[i + 4])
          + decodeBFloat16(b[i + 5]) * decodeBFloat16(a[i + 5])
          + decodeBFloat16(b[i + 6]) * decodeBFloat16(a[i + 6])
          + decodeBFloat16(b[i + 7]) * decodeBFloat16(a[i + 7]);
    }
    return res;
  }

  /**
   * Lucene's vectorized dot product.
   */
  @Benchmark
  public float luceneVectorizedFloat(SimilarityState state) {
    float[] a = state.a;
    float[] b = state.b;
    int i = 0;
    float res = 0;
    // if the array size is large (> 2x platform vector size), its worth the overhead to vectorize
    if (a.length > 2 * FloatVector.SPECIES_PREFERRED.length()) {
      // vector loop is unrolled 4x (4 accumulators in parallel)
      FloatVector acc1 = FloatVector.zero(FloatVector.SPECIES_PREFERRED);
      FloatVector acc2 = FloatVector.zero(FloatVector.SPECIES_PREFERRED);
      FloatVector acc3 = FloatVector.zero(FloatVector.SPECIES_PREFERRED);
      FloatVector acc4 = FloatVector.zero(FloatVector.SPECIES_PREFERRED);
      int upperBound = FloatVector.SPECIES_PREFERRED.loopBound(a.length - 3 * FloatVector.SPECIES_PREFERRED.length());
      for (; i < upperBound; i += 4 * FloatVector.SPECIES_PREFERRED.length()) {
        FloatVector va = FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, a, i);
        FloatVector vb = FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, b, i);
        acc1 = acc1.add(va.mul(vb));
        FloatVector vc =
            FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, a, i + FloatVector.SPECIES_PREFERRED.length());
        FloatVector vd =
            FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, b, i + FloatVector.SPECIES_PREFERRED.length());
        acc2 = acc2.add(vc.mul(vd));
        FloatVector ve =
            FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, a, i + 2 * FloatVector.SPECIES_PREFERRED.length());
        FloatVector vf =
            FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, b, i + 2 * FloatVector.SPECIES_PREFERRED.length());
        acc3 = acc3.add(ve.mul(vf));
        FloatVector vg =
            FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, a, i + 3 * FloatVector.SPECIES_PREFERRED.length());
        FloatVector vh =
            FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, b, i + 3 * FloatVector.SPECIES_PREFERRED.length());
        acc4 = acc4.add(vg.mul(vh));
      }
      // vector tail: less scalar computations for unaligned sizes, esp with big vector sizes
      upperBound = FloatVector.SPECIES_PREFERRED.loopBound(a.length);
      for (; i < upperBound; i += FloatVector.SPECIES_PREFERRED.length()) {
        FloatVector va = FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, a, i);
        FloatVector vb = FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, b, i);
        acc1 = acc1.add(va.mul(vb));
      }
      // reduce
      FloatVector res1 = acc1.add(acc2);
      FloatVector res2 = acc3.add(acc4);
      res += res1.add(res2).reduceLanes(VectorOperators.ADD);
    }

    for (; i < a.length; i++) {
      res += b[i] * a[i];
    }
    return res;
  }

  private static final IntVector SHIFT_16 = IntVector.broadcast(VectorSpecies.of(int.class, ShortVector.SPECIES_PREFERRED.vectorShape()), 16);
  private static final IntVector MASK_16 = IntVector.broadcast(VectorSpecies.of(int.class, ShortVector.SPECIES_PREFERRED.vectorShape()), 0xFFFF0000);

  /**
   * Emulation of a vectorized dot product on bfloat16, by leveraging shifts and masks to convert bfloat16s to floats, and then computing the dot product on floats.
   */
  @Benchmark
  public float vectorizedBFloat16Emulation(SimilarityState state) {
    short[] a = state.a16;
    short[] b = state.b16;

    int upperBound = ShortVector.SPECIES_PREFERRED.loopBound(a.length);
    FloatVector acc = FloatVector.zero(VectorSpecies.of(float.class, ShortVector.SPECIES_PREFERRED.vectorShape()));
    int i;
    for (i = 0; i < upperBound; i += ShortVector.SPECIES_PREFERRED.length()) {
      IntVector aNext = ShortVector.fromArray(ShortVector.SPECIES_PREFERRED, a, i).reinterpretAsInts();
      // FloatVector that stores every bfloat16 at an even index
      FloatVector aNextEven = aNext
          .lanewise(VectorOperators.LSHL, SHIFT_16)
          .reinterpretAsFloats();
      // FloatVector that stores every bfloat16 at an odd index
      FloatVector aNextOdd = aNext
          .and(MASK_16)
          .reinterpretAsFloats();

      IntVector bNext = ShortVector.fromArray(ShortVector.SPECIES_PREFERRED, b, i).reinterpretAsInts();
      FloatVector bNextEven = bNext
          .lanewise(VectorOperators.LSHL, SHIFT_16)
          .reinterpretAsFloats();
      FloatVector bNextOdd = bNext
          .and(MASK_16)
          .reinterpretAsFloats();

      acc = acc.add(aNextEven.mul(bNextEven).add(aNextOdd.mul(bNextOdd)));
    }

    float res = acc.reduceLanes(VectorOperators.ADD);
    for (; i < a.length; ++i) {
      res += decodeBFloat16(a[i]) * decodeBFloat16(b[i]);
    }
    return res;
  }

}
