public class LinearClassifierRunner {

    public static void main(String[] args) {
        int dimNum = 819;
        int numPasses = 5;
        String trainFile = "pa3train.txt";

        LinearClassifier lc = new LinearClassifier(dimNum, trainFile);
        lc.runPerceptronPasses(numPasses);

        print("todo: finish LinearClassifierRunner main");

    }

    private static void print(String text) {
        System.out.println(text);
    }

    


}