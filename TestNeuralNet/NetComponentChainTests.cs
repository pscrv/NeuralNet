using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;

using NeuralNet;
using System.Collections.Generic;

namespace TestNeuralNet
{
    [TestClass]
    public class NetComponentChainTests
    {
        [TestMethod]
        public void CanMake()
        {
            NetComponentChain layerlist = new NetComponentChain();
            Assert.IsNotNull(layerlist);
        }

        [TestMethod]
        public void CanMakeWithContent()
        {
            Layer layer = new Layer(new NeuralNet.WeightsMatrix( new double[,] { { 1 } }) );
            NetComponentChain layerlist = new NetComponentChain(layer);
            Assert.IsNotNull(layerlist);
        }

        [TestMethod]
        public void CanAddFixed()
        {
            NeuralFunction nf = new NeuralFunction(1);
            NetComponentChain layerlist = new NetComponentChain();
            layerlist.AddFixed(nf);
            List<NetComponent> allComponents = new List<NetComponent>(layerlist.ForwardEnumeration);
            List<NetComponent> trainableComponents = new List<NetComponent>(layerlist.ForwardTrainableComponentsEnumeration);
            Assert.AreEqual(1, layerlist.NumberOfComponents);
            Assert.IsTrue(allComponents.Contains(nf));
            Assert.IsFalse(trainableComponents.Contains(nf));
        }

        [TestMethod]
        public void CanAddTrainable()
        {
            WeightedCombiner wc = new WeightedCombiner(new NeuralNet.WeightsMatrix(1, 1));
            NetComponentChain layerlist = new NetComponentChain();
            layerlist.AddTrainable(wc);
            List<NetComponent> allComponents = new List<NetComponent>(layerlist.ForwardEnumeration);
            List<NetComponent> trainableComponents = new List<NetComponent>(layerlist.ForwardTrainableComponentsEnumeration);
            Assert.AreEqual(1, layerlist.NumberOfComponents);
            Assert.IsTrue(allComponents.Contains(wc));
            Assert.IsTrue(trainableComponents.Contains(wc));
        }

        [TestMethod]
        public void CannotAddLayerOfWrongSize()
        {
            Layer layer1 = Layer.CreateLinearLayer(new NeuralNet.WeightsMatrix( new double[,] { { 1 } }) );
            Layer layer2 = Layer.CreateLogisticLayer(new NeuralNet.WeightsMatrix( new double[,] { { 1, 2 } }) );
            NetComponentChain layerlist = new NetComponentChain(layer1);

            try
            {
                layerlist.AddTrainable(layer2);
                Assert.Fail("Add should throw and ArgumentException if when trying to add a layer of the wrong size, but did not.");
            }
            catch (ArgumentException)
            { }
        }

        [TestMethod]
        public void UnrunNetworkHasZeroOutput()
        {
            Layer layer = Layer.CreateLinearLayer(new NeuralNet.WeightsMatrix(new double[,] { { 1 }, { 2 }, { 3 } }));
            NetComponentChain layerlist = new NetComponentChain(layer);
            NeuralNet.NetworkVector outputCheck = new NeuralNet.NetworkVector(new double[] { 0, 0, 0 });
            Assert.AreEqual(outputCheck, layerlist.Output);
        }

        [TestMethod]
        public void CannotRunWithInputOfWrongSize()
        {
            Layer layer = Layer.CreateLinearLayer(new NeuralNet.WeightsMatrix( new double[,] { { 1 } }) );
            NetComponentChain layerlist = new NetComponentChain(layer);
            NeuralNet.NetworkVector input = new NeuralNet.NetworkVector(new double[] { 0, 0 });

            try
            {
                layerlist.Run(input);
                Assert.Fail("Run should throw an ArgumentException for input of the wrong size, but did not.");
            }
            catch (ArgumentException)
            { }
        }

        [TestMethod]
        public void CanRunWithZeroInput()
        {
            Layer layer = Layer.CreateLinearLayer(new NeuralNet.WeightsMatrix( new double[,] { { 1 } }) );
            NetComponentChain layerlist = new NetComponentChain(layer);
            NeuralNet.NetworkVector vector = new NeuralNet.NetworkVector(new double[] { 0 });
            layerlist.Run(vector);
            Assert.AreEqual(vector, layerlist.Output);
        }

        [TestMethod]
        public void CanRunTwoLayersWithZeroInput()
        {
            Layer layer1 = new Layer(new NeuralNet.WeightsMatrix( new double[,] { { 1, 1 }, { 1, 1 } } ));
            Layer layer2 = Layer.CreateLinearLayer(new NeuralNet.WeightsMatrix( new double[,] { { 1, 1 } } ));
            NetComponentChain layerlist = new NetComponentChain(layer1);
            layerlist.AddFixed(layer2);
            NeuralNet.NetworkVector vector = new NeuralNet.NetworkVector(new double[] { 0, 0 });
            layerlist.Run(vector);


            NeuralNet.NetworkVector outputCheck = new NeuralNet.NetworkVector(new double[] { 0 });
            Assert.AreEqual(outputCheck, layerlist.Output);
        }

        [TestMethod]
        public void CanRunTwoLayerNetWithOneInput()
        {
            Layer inputlayer = new Layer(new NeuralNet.WeightsMatrix(new double[,] { { 1, 1, 1 }, { 1, 1, 1 } }));
            Layer outputlayer = Layer.CreateLinearLayer(new NeuralNet.WeightsMatrix(new double[,] { { 1, 1 } }));
            NetComponentChain network = new NetComponentChain();
            network.AddFixed(inputlayer);
            network.AddTrainable(outputlayer);

            NeuralNet.NetworkVector inputvector = new NeuralNet.NetworkVector(new double[] { 1, 0, 0 });
            network.Run(inputvector);

            NeuralNet.NetworkVector outputCheck = new NeuralNet.NetworkVector(new double[] { 2 });
            Assert.AreEqual(outputCheck, network.Output);
        }

        [TestMethod]
        public void CanBackPropagateTwoLayerNetGradient1()
        {
            Layer inputlayer = new Layer(new NeuralNet.WeightsMatrix(new double[,] { { 1, 1, 1 }, { 1, 1, 1 } }));
            Layer outputlayer = Layer.CreateLinearLayer(new NeuralNet.WeightsMatrix(new double[,] { { 1, 1 } }));
            NetComponentChain network = new NetComponentChain();
            network.AddFixed(inputlayer);
            network.AddTrainable(outputlayer);

            NeuralNet.NetworkVector inputvector = new NeuralNet.NetworkVector(new double[] { 1, 0, 0 });
            NeuralNet.NetworkVector outputgradient = new NeuralNet.NetworkVector(new double[] { 1 });

            network.Run(inputvector);

            NeuralNet.NetworkVector inputGradientCheck = new NeuralNet.NetworkVector(new double[] { 2, 2, 2 });
            Assert.AreEqual(inputGradientCheck, network.InputGradient(outputgradient));
        }


    }
}
