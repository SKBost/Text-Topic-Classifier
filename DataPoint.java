public class DataPoint {
    private double[] features;
    private double label;

    public DataPoint(double[] newFeatures, double newLabel) { // todo: see if this is even how you type an array
        features = newFeatures;
        label = newLabel;
    }

    public double[] getFeatures() {
        return features;
    }

    public double getLabel() {
        return label;
    }

    
    //public void setFeatures(Array<int> newFeatures) {
   //     features = newFeatures;
   // }

}
