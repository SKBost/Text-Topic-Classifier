public class LinearClassifier {

    private final int numDims;
    private final int numTrPts; // number of training points
    private final DataPoint[] trPts; // training data points with feature vectors and labels

    public LinearClassifier(int newNumDims, String trainFile) {

        numDims = newNumDims;
        print("todo: read in files");
        print("todo: get actual number of points");
        numTrPts = 8008;
        trPts = new DataPoint[numTrPts];

        print("todo: finish LinearClassifier constructor");

        

    }

    private static double[] multiplyScalarAndVector(double scalar, double[] vector) {
        print("inside multiplyScalarAndVector");
        double[] newVec = new double[vector.length];
        for(int i = 0; i < vector.length; i++) {
            newVec[i] = vector[i] * scalar;
        }
        print("multiplying scalar " + scalar  + "and the following vector:");
        print("");
        printDoubleArray(vector);
        print("multiplied vector is ");
        print("");
        printDoubleArray(newVec);
        return newVec;
    }

    // runs a single pass of perceptron given a starting value
    public double[] runPerceptron(double[] w0) { // note: we start at w0 instead of w1 for 0 indexing
        print("inside runPerceptron");
        double[][] w = new double[numDims][numTrPts + 1]; // todo: do we actually use numPoints + 1?
        w[0] = w0;

        for(int t = 0; t < numTrPts; t++) {
            double yt = trPts[t].getLabel();
            double[] xt  = trPts[t].getFeatures();
            double classVal = yt * getDotProduct(w[t], xt); // classification value

            if(classVal <= 0) {
                w[t+1] = sumVectors(w[t], multiplyScalarAndVector(yt, xt));
            } else {
                w[t+1] = w[t];
            }
        }

        return w[numTrPts]; // final form of w after running on last data point
    }

    private double[] sumVectors(double[] first, double[] second) {
        print("inside sumVectors");

        double[] newVec = new double[first.length];
        
        if(first.length != second.length) {
            print("ERROR in sumVectors: first.length != second.length; returning placeholder zero vector");
        } else {
            for(int i = 0; i < first.length; i++) {
                newVec[i] = first[i] + second[i];
            }
        }

        return newVec;
    }

    private double getDotProduct(double[] wt, double[] xt) {
        print("inside getDotProduct");
        // check if inputs are valid
        if(wt.length != xt.length) {
            print("ERROR in getDotProduct: wt.length != xt.length; returning placeholder 8310094");
            return 8310094;
        }

        // get inner/dot product
        double multDimSum = 0;
        for(int i = 0; i < wt.length; i++) {
            multDimSum += wt[i] * xt[i];
            print("multDimSum (now " + multDimSum + ") = previous multDimSum + wt[" + i +"] (" + wt[i] + ") * xt[" + i + "](" + xt[i] + ")");
        }
        return multDimSum;
    }

    /*
    // this method might be trivial
    private double[] getZeroVector() {
        double[] zeroVec = new double[dimNum];
        return zeroVec;
    }
    */


    public void runPerceptronPasses(int numPasses) {
        print("inside runPerceptronPasses");
        double[] w0 = new double[numDims]; // starter values of 0 for perceptron to build off
        for(int i = 0; i < numPasses; i++) {
            w0 = runPerceptron(w0); // the final value/output of each pass is the starter for the next pass
            print("after run " + i + "of perceptron, the output is the following:");
            print("");
            printDoubleArray(w0);
        }
    }

    public static void printDoubleArray(double[] array) {
        System.out.print("[");
            for(int j = 0; j < array.length; j++) {
                System.out.print(array[j] + ", ");
            }
            System.out.println("]");
    }


    private static void print(String text) {
        System.out.println(text);
    }
}