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

            

            InputLayer = Layer.CreateLinearLayer(new NetworkMatrix(inputWeights), new NetworkVector(inputneurons));
            OutputLayer = Layer.CreateLinearLayer(new NetworkMatrix(outputWeights), new NetworkVector(outputneurons));
        }

        public void Run(NetworkVector input)
        {
            InputLayer.Run(input);
            OutputLayer.Run(InputLayer.Output);
        }

        public NetworkVector InputGradient(NetworkVector outputgradient)
        {
            return InputLayer.InputGradient(OutputLayer.InputGradient(outputgradient));
        }



        #region tryout
        public IEnumerable<StateGradient> GradientBackPropagator(NetworkVector outputgradient)
        {
            List<NetComponent> components = new List<NetComponent> { OutputLayer, InputLayer };

            NetworkVector gradient = outputgradient;
            NetworkVector oldGradient;
            foreach (NetComponent component in components)
            {
                if (component is TrainableComponent)
                {
                    // need to get gradient before yield, because
                    // state of component is likely to change between yields
                    oldGradient = gradient;
                    gradient = component.InputGradient(gradient);
                    yield return (component as TrainableComponent).GetStateGradient(oldGradient);
                }

            }
        }
        #endregion

    }
}
