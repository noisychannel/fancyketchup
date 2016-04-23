import sys
import theano
import cPickle

from train import load_data


def predict(dataset, model_file):
    """
    Loads a model file and predicts on the test set
    """
    classifier = cPickle.load(open(model_file))
    predict_model = theano.function(inputs=[classifier.input],
                                    outputs=classifier.y_pred)
    datasets = load_data(dataset)
    test_set_x, test_set_y = datasets[2]
    test_set_x = test_set_x.get_value()

    predicted_values = predict_model(test_set_x)
    print("Predicted values for the first 10 examples in the test test:")
    print(predicted_values[:10])

    test_model = theano.function(inputs=[classifier.input],
                                 outputs=classifier.errors(test_set_y))
    test_error = test_model(test_set_x)
    print("Test error is %f %%" % (test_error * 100))

if __name__ == '__main__':
    predict(sys.argv[1], sys.argv[2])
