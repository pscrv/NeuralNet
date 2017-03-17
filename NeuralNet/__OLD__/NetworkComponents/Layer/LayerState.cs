using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNet
{
    public class LayerState
    {
        public double[,] Weights { get; private set; }
        public double[] Biases { get; private set; }

        public LayerState(NetworkMatrix weights, NetworkVector biases)
        {
            Biases = biases.ToArray();
            Weights = weights.ToArray();
        }
    }
}
