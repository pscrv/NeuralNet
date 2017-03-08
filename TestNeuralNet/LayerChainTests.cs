using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;

using NeuralNet;

namespace TestNeuralNet
{
    [TestClass]
    public class LayerChainTests
    {
        [TestMethod]
        public void CanMakeLayerList()
        {
            LayerChain layerlist = new LayerChain();
            Assert.AreNotEqual(null, layerlist);
        }


        [TestMethod]
        public void CanMakeLayerListWithContent()
        {
            Layer layer = new Layer(new double[,] { { 1 } });
            LayerChain layerlist = new LayerChain(layer); 
            Assert.AreNotEqual(null, layerlist);
        }

        [TestMethod]
        public void CanCountLayers()
        {            
            Layer layer = new Layer(new double[,] { { 1 } });

            LayerChain layerlist = new LayerChain();
            Assert.AreEqual(0, layerlist.NumberOfLayers);

            for (int i = 1; i < 10; i++)
            {
                layerlist.Add(layer);
                Assert.AreEqual(i, layerlist.NumberOfLayers);
            }
        }

        [TestMethod]
        public void CannotAddLayerOfWrongSize()
        {
            Layer layer1 = new Layer(new double[,] { { 1 } });
            Layer layer2 = new Layer(new double[,] { { 1, 2 } });
            LayerChain layerlist = new LayerChain(layer1);

            try
            {
                layerlist.Add(layer2);
                Assert.Fail("Add should throw and ArgumentException if when trying to add a layer of the wrong size, but did not.");
            }
            catch(ArgumentException)
            { }

            Assert.AreNotEqual(null, layerlist);
        }
        
        [TestMethod]
        public void UnrunNetworkHasZeroOutput()
        {
            Layer layer = new Layer(new double[,] { { 1 }, { 2 }, { 3 } });
            LayerChain layerlist = new LayerChain(layer);
            NetworkVector outputCheck = new NetworkVector( new double[] { 0, 0, 0 } );
            Assert.AreEqual(outputCheck, layerlist.Output);
        }

        [TestMethod]
        public void CannotRunWithInputOfWrongSize()
        {
            Layer layer = new Layer(new double[,] { { 1 } });
            LayerChain layerlist = new LayerChain(layer);
            NetworkVector input = new NetworkVector(new double[] { 0, 0 });

            try
            {
                layerlist.Run(input);
                Assert.Fail("Run should throw an ArgumentException for input of the wrong size, but did not.");
            }
            catch (ArgumentException)
            { }
        }

        [TestMethod]
        public void CannotRunWithZeroInput()
        {
            Layer layer = new Layer(new double[,] { { 1 } });
            LayerChain layerlist = new LayerChain(layer);
            NetworkVector vector = new NetworkVector(new double[] { 0 });
            layerlist.Run(vector);
            Assert.AreEqual(vector, layerlist.Output);            
        }

        [TestMethod]
        public void CanRunTwoLayersWithZeroInput()
        {
            Layer layer1 = new Layer(new double[,] { { 1, 1 }, { 1, 1 } });
            Layer layer2 = new Layer(new double[,] { { 1, 1 } });
            LayerChain layerlist = new LayerChain(layer1);
            layerlist.Add(layer2);
            NetworkVector vector = new NetworkVector(new double[] { 0, 0 });
            layerlist.Run(vector);
            Assert.AreEqual(0, layerlist.Output.ToArray()[0]);
        }

        [TestMethod]
        public void CanRunTwoLayerNetWithOneInput()
        {
            Layer inputlayer = new Layer(new double[,] { { 1, 1, 1 }, { 1, 1, 1 } });
            Layer outputlayer = new Layer(new double[,] { { 1, 1 } });
            LayerChain network = new LayerChain();
            network.Add(inputlayer);
            network.Add(outputlayer);

            NetworkVector inputvector = new NetworkVector(new double[] { 1, 0, 0 });
            network.Run(inputvector);

            double output = network.Output.ToArray()[0];
            Assert.AreEqual(2, output);
        }

        [TestMethod]
        public void CanBackPropagateTwoLayerNetGradient1()
        {
            Layer inputlayer = new Layer(new double[,] { { 1, 1, 1 }, { 1, 1, 1 } });
            Layer outputlayer = new Layer(new double[,] { { 1, 1 } });
            LayerChain network = new LayerChain();
            network.Add(inputlayer);
            network.Add(outputlayer);

            NetworkVector inputvector = new NetworkVector(new double[] { 1, 0, 0 });
            NetworkVector outputgradient = new NetworkVector(new double[] { 1 });

            network.Run(inputvector);
            network.BackPropagate(outputgradient);

            LayerState inputState = network.State[0];
            LayerState outputState = network.State[1];

            
            double[,] inputWeights = inputState.Weights;
            double[,] inputWeightsCheck = new double[,] { { 0, 1, 1 }, { 0, 1, 1 } };
            double[] inputBiases = inputState.Biases;
            double[] inputBiasesCheck = new double[] { -1, -1 };
            for (int i = 0; i < inputWeights.GetLength(0); i++)
            {
                Assert.AreEqual(inputBiasesCheck[i], inputBiases[i]);

                for (int j = 0; j < inputWeights.GetLength(1); j++)
                    Assert.AreEqual(inputWeights[i, j], inputWeights[i, j]);
            }
            
            double[,] outputWeights = outputState.Weights;
            double[,] outputWeightsCheck = new double[,] { { 0, 0 } };
            double[] outputBiases = outputState.Biases;
            double[] outputBiasesCheck = new double[] { -1 };
            for (int i = 0; i < outputWeights.GetLength(0); i++)
            {
                Assert.AreEqual(outputBiasesCheck[i], outputBiases[i]);
                for (int j = 0; j < outputWeights.GetLength(1); j++)
                    Assert.AreEqual(outputWeightsCheck[i, j], outputWeights[i, j]);
            }

        }

    }
}
