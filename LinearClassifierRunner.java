import java.util.*; 
import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.stream.Stream;

public class LinearClassifierRunner {

    public static void main(String[] args) {
        int dimNum = 819;
        int numPasses = 1;
        String trainFile = "pa3train.txt";

        LinearClassifier lc = new LinearClassifier(dimNum, trainFile);
        lc.runPerceptronPasses(numPasses);

        print("todo: finish LinearClassifierRunner main");

    }

    private static void print(String text) {
        System.out.println(text);
    }

    


}