using Microsoft.VisualStudio.TestTools.UnitTesting;

using NeuralNet;

namespace TestNeuralNet
{
    [TestClass]
    public class TestNetworkTests
    {  
        [TestMethod]
        public void CanMakeSmallNet()
        {
            int inputs = 3;
            int inputneurons = 2;
            int outputneurons = 1;

            LinearTwoLayerTestNetwork network = new LinearTwoLayerTestNetwork(inputs, inputneurons, outputneurons);
            Assert.AreNotEqual(null, network);
        }

        [TestMethod]
        public void UnrunSmallNetworkHasZeroOutput()
        {
            int inputs = 3;
            int inputneurons = 2;
            int outputneurons = 1;

            LinearTwoLayerTestNetwork network = new LinearTwoLayerTestNetwork(inputs, inputneurons, outputneurons);
            double output = network.Output.ToArray()[0];
            Assert.AreEqual(0, output);
        }

        [TestMethod]
        public void CanRunSmallNetWithZeroInput()
        {
            int inputs = 3;
            int inputneurons = 2;
            int outputneurons = 1;
            NetworkVector inputvector = new NetworkVector(3);

            LinearTwoLayerTestNetwork network = new LinearTwoLayerTestNetwork(inputs, inputneurons, outputneurons);
            network.Run(inputvector);

            double output = network.Output.ToArray()[0];
            Assert.AreEqual(0, output);
        }

        [TestMethod]
        public void CanRunSmallNetWithOneInput()
        {
            int inputs = 3;
            int inputneurons = 2;
            int outputneurons = 1;
            NetworkVector inputvector = new NetworkVector(new double[] { 1, 0, 0 });

            LinearTwoLayerTestNetwork network = new LinearTwoLayerTestNetwork(inputs, inputneurons, outputneurons);
            network.Run(inputvector);

            double output = network.Output.ToArray()[0];
            Assert.AreEqual(2, output);
        }

        [TestMethod]
        public void CanRunSmallNetWithTwoInputs()
        {
            int inputs = 3;
            int inputneurons = 2;
            int outputneurons = 1;
            NetworkVector inputvector = new NetworkVector(new double[] { 0, 1, 1 });

            LinearTwoLayerTestNetwork network = new LinearTwoLayerTestNetwork(inputs, inputneurons, outputneurons);
            network.Run(inputvector);

            double output = network.Output.ToArray()[0];
            Assert.AreEqual(4, output);
        }

        [TestMethod]
        public void CanRunSmallNetWithThreeInputs()
        {
            int inputs = 3;
            int inputneurons = 2;
            int outputneurons = 1;
            NetworkVector inputvector = new NetworkVector(new double[] { 1, 1, 1 });

            LinearTwoLayerTestNetwork network = new LinearTwoLayerTestNetwork(inputs, inputneurons, outputneurons);
            network.Run(inputvector);

            double output = network.Output.ToArray()[0];
            Assert.AreEqual(6, output);
        }


        [TestMethod]
        public void CanBackPropagateSmallNetGradient1()
        {
            int inputs = 3;
            int inputneurons = 2;
            int outputneurons = 1;
            NetworkVector inputvector = new NetworkVector(new double[] { 1, 0, 0 });
            NetworkVector outputgradient = new NetworkVector(new double[] { 1 });

            LinearTwoLayerTestNetwork network = new LinearTwoLayerTestNetwork(inputs, inputneurons, outputneurons);
            network.Run(inputvector);
            network.BackPropagate(outputgradient);

            LayerState inputState = network.InputLayer.State;
            double[,] inputWeights = inputState.Weights;
            double[,] inputWeightsCheck = new double[,] { { 0, 1, 1 }, { 0, 1, 1 } };
            double[] inputBiases = inputState.Biases;
            double[] inputBiasesCheck = new double[] { -1, -1 };
            for (int i = 0; i < inputneurons; i++)
            {
                Assert.AreEqual(inputBiasesCheck[i], inputBiases[i]);

                for (int j = 0; j < inputs; j++)
                    Assert.AreEqual(inputWeights[i, j], inputWeights[i, j]);
            }

            LayerState outputState = network.OutputLayer.State;
            double[,] outputWeights = outputState.Weights;
            double[,] outputWeightsCheck = new double[,] { { 0, 0 } };
            double[] outputBiases = outputState.Biases;
            double[] outputBiasesCheck = new double[] { -1 };
            for (int i = 0; i < outputneurons; i++)
            {
                Assert.AreEqual(outputBiasesCheck[i], outputBiases[i]);
                for (int j = 0; j < inputneurons; j++)
                    Assert.AreEqual(outputWeightsCheck[i, j], outputWeights[i, j]);
            }
        }


        [TestMethod]
        public void CanBackPropagateSmallNetGradient0()
        {
            int inputs = 3;
            int inputneurons = 2;
            int outputneurons = 1;
            NetworkVector inputvector = new NetworkVector(new double[] { 1, 0, 0 });
            NetworkVector outputgradient = new NetworkVector(new double[] { 0 });

            LinearTwoLayerTestNetwork network = new LinearTwoLayerTestNetwork(inputs, inputneurons, outputneurons);
            network.Run(inputvector);
            network.BackPropagate(outputgradient);

            LayerState inputState = network.InputLayer.State;
            double[,] inputWeights = inputState.Weights;
            double[,] inputWeightsCheck = new double[,] { { 1, 1, 1 }, { 1, 1, 1 } };
            double[] inputBiases = inputState.Biases;
            double[] inputBiasesCheck = new double[] { 0, 0 };
            for (int i = 0; i < inputneurons; i++)
            {
                Assert.AreEqual(inputBiasesCheck[i], inputBiases[i]);

                for (int j = 0; j < inputs; j++)
                    Assert.AreEqual(inputWeights[i, j], inputWeights[i, j]);
            }

            LayerState outputState = network.OutputLayer.State;
            double[,] outputWeights = outputState.Weights;
            double[,] outputWeightsCheck = new double[,] { { 1, 01} };
            double[] outputBiases = outputState.Biases;
            double[] outputBiasesCheck = new double[] { 0 };
            for (int i = 0; i < outputneurons; i++)
            {
                Assert.AreEqual(outputBiasesCheck[i], outputBiases[i]);
                for (int j = 0; j < inputneurons; j++)
                    Assert.AreEqual(outputWeightsCheck[i, j], outputWeights[i, j]);
            }
        }

        [TestMethod]
        public void CanBackPropagateSmallNetGradientThird()
        {
            int inputs = 3;
            int inputneurons = 2;
            int outputneurons = 1;
            NetworkVector inputvector = new NetworkVector(new double[] { 1, 0, 0 });
            NetworkVector outputgradient = new NetworkVector(new double[] { 1.0/3 });

            LinearTwoLayerTestNetwork network = new LinearTwoLayerTestNetwork(inputs, inputneurons, outputneurons);
            network.Run(inputvector);
            network.BackPropagate(outputgradient);

            LayerState inputState = network.InputLayer.State;
            double[,] inputWeights = inputState.Weights;
            double[,] inputWeightsCheck = new double[,] { { 2.0/3, 1, 1 }, { 2.0/3, 1, 1 } };
            double[] inputBiases = inputState.Biases;
            double[] inputBiasesCheck = new double[] { -1.0/3, -1.0/3 };
            for (int i = 0; i < inputneurons; i++)
            {
                Assert.AreEqual(inputBiasesCheck[i], inputBiases[i], 0.00000001);

                for (int j = 0; j < inputs; j++)
                    Assert.AreEqual(inputWeights[i, j], inputWeights[i, j], 0.00000001);
            }

            LayerState outputState = network.OutputLayer.State;
            double[,] outputWeights = outputState.Weights;
            double[,] outputWeightsCheck = new double[,] { { 2.0/3, 2.0/3 } };
            double[] outputBiases = outputState.Biases;
            double[] outputBiasesCheck = new double[] { -1.0/3 };
            for (int i = 0; i < outputneurons; i++)
            {
                Assert.AreEqual(outputBiasesCheck[i], outputBiases[i], 0.00000001);

                for (int j = 0; j < inputneurons; j++)
                    Assert.AreEqual(outputWeightsCheck[i, j], outputWeights[i, j], 0.00000001);
            }
        }







        [TestMethod]
        public void CanMakeBigNet()
        {
            int inputs = 100;
            int inputneurons = 2000;
            int outputneurons = 1;

            LinearTwoLayerTestNetwork network = new LinearTwoLayerTestNetwork(inputs, inputneurons, outputneurons);
            Assert.AreNotEqual(null, network);
        }

        [TestMethod]
        public void UnrunBigNetworkHasZeroOutput()
        {
            int inputs = 100;
            int inputneurons = 2000;
            int outputneurons = 1;

            LinearTwoLayerTestNetwork network = new LinearTwoLayerTestNetwork(inputs, inputneurons, outputneurons);
            double output = network.Output.ToArray()[0];
            Assert.AreEqual(0, output);
        }

        [TestMethod]
        public void CanRunBigNetWithZeroInput()
        {
            int inputs = 100;
            int inputneurons = 2000;
            int outputneurons = 1;
            NetworkVector inputvector = new NetworkVector(100);

            LinearTwoLayerTestNetwork network = new LinearTwoLayerTestNetwork(inputs, inputneurons, outputneurons);
            network.Run(inputvector);

            double output = network.Output.ToArray()[0];
            Assert.AreEqual(0, output);
        }

        [TestMethod]
        public void CanRunBigNetWithOneInput()
        {
            int inputs = 100;
            int inputneurons = 2000;
            int outputneurons = 1;
            double[] inputArray = new double[inputs];
            inputArray[0] = 1;
            NetworkVector inputvector = new NetworkVector(inputArray);

            LinearTwoLayerTestNetwork network = new LinearTwoLayerTestNetwork(inputs, inputneurons, outputneurons);
            network.Run(inputvector);

            double output = network.Output.ToArray()[0];
            Assert.AreEqual(2000, output);
        }

        //here
        [TestMethod]
        public void CanRunBigNetWithTwoInputs()
        {
            int inputs = 100;
            int inputneurons = 2000;
            int outputneurons = 1;
            double[] inputArray = new double[inputs];
            inputArray[0] = 1;
            inputArray[5] = 1;                
            NetworkVector inputvector = new NetworkVector(inputArray);

            LinearTwoLayerTestNetwork network = new LinearTwoLayerTestNetwork(inputs, inputneurons, outputneurons);
            network.Run(inputvector);

            double output = network.Output.ToArray()[0];
            Assert.AreEqual(4000, output);
        }

        [TestMethod]
        public void CanRunBigNetWithThreeInputs()
        {
            int inputs = 100;
            int inputneurons = 2000;
            int outputneurons = 1;
            double[] inputArray = new double[inputs];
            inputArray[0] = 1;
            inputArray[7] = 1;
            inputArray[77] = 1;
            NetworkVector inputvector = new NetworkVector(inputArray);

            LinearTwoLayerTestNetwork network = new LinearTwoLayerTestNetwork(inputs, inputneurons, outputneurons);
            network.Run(inputvector);

            double output = network.Output.ToArray()[0];
            Assert.AreEqual(6000, output);
        }

        [TestMethod]
        public void CanBackPropagateBigNetGradient1()
        {
            int inputs = 100;
            int inputneurons = 2000;
            int outputneurons = 1;
            double[] inputArray = new double[inputs];
            inputArray[0] = 1;
            NetworkVector inputvector = new NetworkVector(inputArray);
            NetworkVector outputgradient = new NetworkVector(new double[] { 1 });

            LinearTwoLayerTestNetwork network = new LinearTwoLayerTestNetwork(inputs, inputneurons, outputneurons);
            network.Run(inputvector);
            network.BackPropagate(outputgradient);

            LayerState inputState = network.InputLayer.State;
            LayerState outputState = network.OutputLayer.State;

            int check;
            for (int i = 0; i < inputneurons; i++)
            {
                Assert.AreEqual(-1, inputState.Biases[i]);

                for (int j = 0; j < inputs; j++)
                {
                    check = j == 0 ? 0: 1;
                    Assert.AreEqual(check, inputState.Weights[i, j]);
                }
            }
            
            for (int i = 0; i < outputneurons; i++)
            {
                Assert.AreEqual(-1, outputState.Biases[i]);

                for (int j = 0; j < inputneurons; j++)
                {
                    Assert.AreEqual(0, outputState.Weights[i, j]);
                }
            }

        }


    }
}
