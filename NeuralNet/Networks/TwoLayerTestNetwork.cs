using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNet
{
    public class LinearTwoLayerTestNetwork
    {
        public Layer InputLayer;
        public Layer OutputLayer;

        public NetworkVector Output { get { return OutputLayer.Output; } }

        public LinearTwoLayerTestNetwork(int inputs, int inputneurons, int outputneurons)
        {
            double[,] inputWeights = new double[inputneurons, inputs];
            double[,] outputWeights = new double[outputneurons, inputneurons];

            for (int i = 0; i < inputneurons; i++)
                for (int j = 0; j < inputs; j++)
                    inputWeights[i, j] = 1;

            for (int i = 0; i < outputneurons; i++)
                for (int j = 0; j < inputneurons; j++)
                    outputWeights[i, j] = 1;

            InputLayer = new LinearLayer(inputWeights);
            OutputLayer = new LinearLayer(outputWeights);
        }

        public void Run(NetworkVector input)
        {
            InputLayer.Run(input);
            OutputLayer.Run(InputLayer.Output);
        }

        public void BackPropagate(NetworkVector outputgradient)
        {
            OutputLayer.BackPropagate(outputgradient);
            InputLayer.BackPropagate(OutputLayer.InputGradient);
        }

    }
}
