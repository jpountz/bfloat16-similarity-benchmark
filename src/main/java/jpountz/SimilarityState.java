package jpountz;

import org.openjdk.jmh.annotations.Level;
import org.openjdk.jmh.annotations.Param;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;

@State(Scope.Benchmark)
public class SimilarityState {

  @Param({"384", "1024"})
  int size;

  float[] a, b;
  short[] a16, b16;

  @Setup(Level.Trial)
  public void setup() {
    a = new float[size];
    b = new float[size];
    for (int i = 0; i < size; ++i) {
      a[i] = (float) i / (i + 1);
      b[i] = 1f / (i + 1);
    }

    a16 = new short[size];
    b16 = new short[size];
    for (int i = 0; i < size; ++i) {
      a16[i] = floatToBFloat16(a[i]);
      b16[i] = floatToBFloat16(b[i]);
    }
  }

  private static short floatToBFloat16(float f) {
    int bits = Float.floatToIntBits(f);
    // This does not implement rounding correctly but is a good first approximation for this benchmark.
    return (short) ((bits + 0x7FFF) >>> 16);
  }

}
