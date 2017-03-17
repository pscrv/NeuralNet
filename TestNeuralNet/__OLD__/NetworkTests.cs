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
        public void InputGradientSmallNet1()
        {
            int inputs = 3;
            int inputneurons = 2;
            int outputneurons = 1;
            NetworkVector inputvector = new NetworkVector(new double[] { 1, 0, 0 });
            NetworkVector outputgradient = new NetworkVector(new double[] { 1 });

            LinearTwoLayerTestNetwork network = new LinearTwoLayerTestNetwork(inputs, inputneurons, outputneurons);
            network.Run(inputvector);

            NetworkVector inputGradientCheck = new NetworkVector(new double[] { 2, 2, 2 });
            Assert.AreEqual(inputGradientCheck, network.InputGradient(outputgradient));

        }


        [TestMethod]
        public void InputGradientSmallNet0()
        {
            int inputs = 3;
            int inputneurons = 2;
            int outputneurons = 1;
            NetworkVector inputvector = new NetworkVector(new double[] { 1, 0, 0 });
            NetworkVector outputgradient = new NetworkVector(new double[] { 0 });

            LinearTwoLayerTestNetwork network = new LinearTwoLayerTestNetwork(inputs, inputneurons, outputneurons);
            network.Run(inputvector);

            NetworkVector inputGradientCheck = new NetworkVector(new double[] { 0, 0, 0 });
            Assert.AreEqual(inputGradientCheck, network.InputGradient(outputgradient));

        }

        [TestMethod]
        public void InputGradientSmallNetGradientThird()
        {
            int inputs = 3;
            int inputneurons = 2;
            int outputneurons = 1;
            NetworkVector inputvector = new NetworkVector(new double[] { 1, 0, 0 });
            NetworkVector outputgradient = new NetworkVector(new double[] { 1.0 / 3 });

            LinearTwoLayerTestNetwork network = new LinearTwoLayerTestNetwork(inputs, inputneurons, outputneurons);
            network.Run(inputvector);

            NetworkVector inputGradientCheck = new NetworkVector(new double[] { 2.0/3, 2.0/3, 2.0/3 });
            Assert.AreEqual(inputGradientCheck, network.InputGradient(outputgradient));
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
        public void InputGradientBigNetGradient1()
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

            double[] inputGradientCheckValues = new double[inputs];
            for (int i = 0; i < inputs; i++)
            {
                inputGradientCheckValues[i] = inputneurons;
            }

            NetworkVector inputGradientCheck = new NetworkVector(inputGradientCheckValues);
            Assert.AreEqual(inputGradientCheck, network.InputGradient(outputgradient));
        }


    }
}
