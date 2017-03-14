using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;

using NeuralNet;

namespace TestNeuralNet
{
    [TestClass]
    public class BatchWeightedCombinerests
    {
        [TestMethod]
        public void CanMakeBWC()
        {
            NetworkMatrix weights = new NetworkMatrix(new double[,]{ { 1 } });
            BatchWeightedCombiner wc = new BatchWeightedCombiner(weights);
            Assert.AreNotEqual(null, wc);
            Assert.AreEqual(1, wc.NumberOfOutputs);
            Assert.AreEqual(1, wc.NumberOfInputs);
            Assert.AreEqual(1, wc.State.Weights[0, 0]);
            Assert.AreEqual(0, wc.State.Biases[0]);
        }

        [TestMethod]
        public void CanMakeBWCWithBiases()
        {
            NetworkMatrix weights = new NetworkMatrix(new double[,] { { 1 } });
            NetworkVector biases = new NetworkVector(new double[] { 3 } );
            BatchWeightedCombiner wc = new BatchWeightedCombiner(weights, biases);
            Assert.AreNotEqual(null, wc);
            Assert.AreEqual(1, wc.NumberOfOutputs);
            Assert.AreEqual(1, wc.NumberOfInputs);
            Assert.AreEqual(1, wc.State.Weights[0, 0]);
            Assert.AreEqual(3, wc.State.Biases[0]);
        }

        [TestMethod]
        public void CanMakeBWCWithNullBiases()
        {
            NetworkMatrix weights = new NetworkMatrix(new double[,] { { 1 } });
            NetworkVector biases = null;
            try
            {
                BatchWeightedCombiner wc = new BatchWeightedCombiner(weights, biases);
                Assert.Fail("Failure to throw ArgumentException when trying to create a WeightedCombiner with null biases.");
            }
            catch (ArgumentException) { }
        }

        [TestMethod]
        public void CanMakeBWCWithNullWeights()
        {
            NetworkMatrix weights = null;
            NetworkVector biases = new NetworkVector(new double[] { 1 });
            try
            {
                BatchWeightedCombiner wc = new BatchWeightedCombiner(weights, biases);
                Assert.Fail("Failure to throw ArgumentException when trying to create a WeightedCombiner with null weights.");
            }
            catch (ArgumentException) { }
        }

        [TestMethod]
        public void CanRunBWC()
        {
            NetworkMatrix weights = new NetworkMatrix(new double[,] { { 1 } });
            NetworkVector biases = new NetworkVector(new double[] { 3 });
            BatchWeightedCombiner wc = new BatchWeightedCombiner(weights, biases);
            NetworkVector input = new NetworkVector(wc.NumberOfInputs);

            wc.Run(input);
            
            Assert.AreEqual(3, wc.Output.ToArray()[0]);
        }

        [TestMethod]
        public void CanRunBWC_2by3()
        {
            NetworkMatrix weights = new NetworkMatrix(new double[,] { { 1, 2, 3 }, { 5, 7, 11} });
            NetworkVector biases = new NetworkVector(new double[] { 100, 200 });
            BatchWeightedCombiner wc = new BatchWeightedCombiner(weights, biases);
            NetworkVector input = new NetworkVector(new double[] { 1, 2, 3});

            wc.Run(input);

            NetworkVector outputcheck = new NetworkVector(new double[] { 114, 252 });

            Assert.AreEqual(outputcheck, wc.Output);
        }

        [TestMethod]
        public void CanBPWC_trivialBatch()
        {
            NetworkMatrix weights = new NetworkMatrix(new double[,] { { 1 } });
            NetworkVector biases = new NetworkVector(new double[] { 10 });
            BatchWeightedCombiner wc = new BatchWeightedCombiner(weights, biases);
            NetworkVector input = new NetworkVector(new double[] { 1 });
            NetworkVector outputgradient = new NetworkVector(new double[] { 1 });

            wc.StartBatch();
            wc.Run(input);
            wc.BackPropagate(outputgradient);
            wc.EndBatchAndUpdate();

            NetworkVector outputCheck = new NetworkVector(new double[] { 11 });
            NetworkVector inputGradientCheck = new NetworkVector(new double[] { 1 });
            double[,] weightsCheck = new double[,] { { 0 } };
            double[] biasesCheck = new double[] { 9 };
            Assert.AreEqual(outputCheck, wc.Output);
            Assert.AreEqual(inputGradientCheck, wc.InputGradient);

            for (int i = 0; i < wc.NumberOfOutputs; i++)
            {
                Assert.AreEqual(biasesCheck[i], wc.State.Biases[i]);

                for (int j = 0; j < wc.NumberOfInputs; j++)
                {
                    Assert.AreEqual(weightsCheck[i, j], wc.State.Weights[i, j]);
                }

            }
        }

        [TestMethod]
        public void CanBPWC_nonTrivialBatch()
        {
            NetworkMatrix weights = new NetworkMatrix(new double[,] { { 1 } });
            NetworkVector biases = new NetworkVector(new double[] { 10 });
            BatchWeightedCombiner wc = new BatchWeightedCombiner(weights, biases);
            NetworkVector input = new NetworkVector(new double[] { 1 });
            NetworkVector outputgradient = new NetworkVector(new double[] { 1 });

            wc.StartBatch();
            for (int i = 0; i < 2; i++)
            {
                wc.Run(input);
                wc.BackPropagate(outputgradient);
            }
            wc.EndBatchAndUpdate();

            NetworkVector outputCheck = new NetworkVector(new double[] { 11 });
            NetworkVector inputGradientCheck = new NetworkVector(new double[] { 1 });
            double[,] weightsCheck = new double[,] { { -1 } };
            double[] biasesCheck = new double[] { 8 };
            Assert.AreEqual(outputCheck, wc.Output);
            Assert.AreEqual(inputGradientCheck, wc.InputGradient);

            for (int i = 0; i < wc.NumberOfOutputs; i++)
            {
                Assert.AreEqual(biasesCheck[i], wc.State.Biases[i]);

                for (int j = 0; j < wc.NumberOfInputs; j++)
                {
                    Assert.AreEqual(weightsCheck[i, j], wc.State.Weights[i, j]);
                }

            }
        }
        [TestMethod]
        public void CanBPWC2x3_nonTrivialBatch()
        {
            NetworkMatrix weights = new NetworkMatrix(new double[,] { { 1, 2, 3 }, { 5, 7, 11 } });
            NetworkVector biases = new NetworkVector(new double[] { 100, 200 });
            BatchWeightedCombiner wc = new BatchWeightedCombiner(weights, biases);
            NetworkVector input = new NetworkVector(new double[] { 1, 2, 3 });
            NetworkVector outputgradient = new NetworkVector(new double[] { 1, 1 });

            wc.StartBatch();
            for (int i = 0; i < 2; i++)
            {
                wc.Run(input);
                wc.BackPropagate(outputgradient);
            }
            wc.EndBatchAndUpdate();

            NetworkVector outputCheck = new NetworkVector(new double[] { 114, 252 });
            NetworkVector inputGradientCheck = new NetworkVector(new double[] { 6, 9, 14 });
            double[,] weightsCheck = new double[,] { { -1 , -2 , -3 }, { 3, 3, 5 } };
            double[] biasesCheck = new double[] { 98, 198 };
            Assert.AreEqual(outputCheck, wc.Output);
            Assert.AreEqual(inputGradientCheck, wc.InputGradient);

            for (int i = 0; i < wc.NumberOfOutputs; i++)
            {
                Assert.AreEqual(biasesCheck[i], wc.State.Biases[i]);

                for (int j = 0; j < wc.NumberOfInputs; j++)
                {
                    Assert.AreEqual(weightsCheck[i, j], wc.State.Weights[i, j]);
                }

            }
        }

    }
}
