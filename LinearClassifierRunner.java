import java.util.*; 
import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.stream.Stream;

public class LinearClassifierRunner {

    public static void main(String[] args) {
        int dimNum = 819;
        int numPasses = 4;
        int numIterations = 100;
        int numIdxs = 3;
        String trainFile = "pa3train.txt";
        String testFile = "pa3test.txt";

        LinearClassifier lc = new LinearClassifier(dimNum, trainFile, testFile);

        lc.useClasses1And2(); // part 1 and part 2 use classes 1 and 2

        // part 1

        print("Part 1:");
        lc.runPerceptronPasses(numPasses);

        // part 2

        print("\nPart 2:");
        lc.runGradientDescent(numIterations);

        // part 3

        print("\nPart 3:");
        lc.getExtremeCoords(lc.getPW3(), numIdxs);

        // part 4

        print("\nPart 4:");
        lc.getExtremeCoords(lc.getLRW50(), numIdxs);


        print("todo: finish LinearClassifierRunner main");

    }

    private static void print(String text) {
        System.out.println(text);
    }

    


}