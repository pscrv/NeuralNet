using Microsoft.VisualStudio.TestTools.UnitTesting;

using NeuralNet;
using System.Collections.Generic;

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
        public void TrainOnline_SmallNet_Correct()
        {
            int inputs = 3;
            int inputneurons = 2;
            int outputneurons = 1;
            
            LinearTwoLayerTestNetwork network = new LinearTwoLayerTestNetwork(inputs, inputneurons, outputneurons);
            List<VectorPair> trainingVectors = new List<VectorPair>
            {
                new VectorPair(
                    new NetworkVector( new double[] { 0, 0, 0 } ),
                    new NetworkVector( new double[] { 1 })
                    ),
                new VectorPair(
                    new NetworkVector( new double[] { 1, 0, 0 } ),
                    new NetworkVector( new double[] { 0 })
                    ),
                new VectorPair(
                    new NetworkVector( new double[] { 0, 1, 0 } ),
                    new NetworkVector( new double[] { 0 })
                    ),
                new VectorPair(
                    new NetworkVector( new double[] { 1, 1, 0 } ),
                    new NetworkVector( new double[] { 1 })
                    )
            };

            NetworkTrainer trainer = new OnlineNetworkTrainer(network, trainingVectors);
            trainer.Train();

            WeightsMatrix inputWeightsCheck = new WeightsMatrix(new double[,] { { -35499715, -35499260, 1 },{ -35499715, -35499260, 1 }});
            NetworkVector inputBiasesCheck = new NetworkVector(new double[] { -35499265, -35499265 });
            WeightsMatrix outputWeightsCheck = new WeightsMatrix(new double[,] { { -224831362, -224831362 } });
            NetworkVector outputBiasesCheck = new NetworkVector(new double[] {-251825});

            Assert.AreEqual(inputWeightsCheck, network.InputLayer.Weights);
            Assert.AreEqual(inputBiasesCheck, network.InputLayer.Biases);
            Assert.AreEqual(outputWeightsCheck, network.OutputLayer.Weights);
            Assert.AreEqual(outputBiasesCheck, network.OutputLayer.Biases);
        }


        [TestMethod]
        public void TrainBatch_SmallNet_Correct()
        {
            int inputs = 3;
            int inputneurons = 2;
            int outputneurons = 1;

            LinearTwoLayerTestNetwork network = new LinearTwoLayerTestNetwork(inputs, inputneurons, outputneurons);
            List<VectorPair> trainingVectors = new List<VectorPair>
            {
                new VectorPair(
                    new NetworkVector( new double[] { 0, 0, 0 } ),
                    new NetworkVector( new double[] { 1 })
                    ),
                new VectorPair(
                    new NetworkVector( new double[] { 1, 0, 0 } ),
                    new NetworkVector( new double[] { 0 })
                    ),
                new VectorPair(
                    new NetworkVector( new double[] { 0, 1, 0 } ),
                    new NetworkVector( new double[] { 0 })
                    ),
                new VectorPair(
                    new NetworkVector( new double[] { 1, 1, 0 } ),
                    new NetworkVector( new double[] { 1 })
                    )
            };

            NetworkTrainer trainer = new BatchNetworkTrainer(network, trainingVectors);
            trainer.Train();

            WeightsMatrix inputWeightsCheck = new WeightsMatrix(new double[,] { { -4, -4, 1 }, { -4, -4, 1 } });
            NetworkVector inputBiasesCheck = new NetworkVector(new double[] { -6, -6 });
            WeightsMatrix outputWeightsCheck = new WeightsMatrix(new double[,] { { -9, -9 } });
            NetworkVector outputBiasesCheck = new NetworkVector(new double[] { -6 });

            Assert.AreEqual(inputWeightsCheck, network.InputLayer.Weights);
            Assert.AreEqual(inputBiasesCheck, network.InputLayer.Biases);
            Assert.AreEqual(outputWeightsCheck, network.OutputLayer.Weights);
            Assert.AreEqual(outputBiasesCheck, network.OutputLayer.Biases);
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
