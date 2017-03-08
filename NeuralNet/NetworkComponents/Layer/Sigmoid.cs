using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNet
{
    public class SigmoidLayer : Layer
    {
        static private double sigmoid(double input)
        {
            return 1.0 / (1 + Math.Exp(-input));
        }

        static private double sigmoidDerivative(double input, double output)
        {
            return output * (1 - output);
        }

        public SigmoidLayer(double[,] weights, double[] biases = null)
            : base(weights:weights, biases:biases, activationfunction:sigmoid, derivativefunction:sigmoidDerivative)
        { }
    }
}
