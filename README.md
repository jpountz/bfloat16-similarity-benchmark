Compare similarity computations between float and bfloat16.

To run benchmarks:

$ mvn clean package

$ java -jar target/benchmarks.jar

To see the produced assembly:

$ java -jar target/benchmarks.jar -prof perfasm
