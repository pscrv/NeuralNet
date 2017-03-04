using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNet
{
    public class LinearLayer : Layer
    {
        public LinearLayer(double[,] weights, double[] biases = null)
            : base(weights, biases)
        { }
    }
}
