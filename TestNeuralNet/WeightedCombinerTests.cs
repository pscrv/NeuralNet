using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;

using NeuralNet;

namespace TestNeuralNet
{
    [TestClass]
    public class WeightedCombinerTests
    {
        WeightedCombiner wc_2;
        WeightedCombiner wc_1;
        WeightedCombiner wc_1b;

        WeightsMatrix matrix_1 = new WeightsMatrix(new double[,] { { 1 } });
        WeightsMatrix matrix_2 = new WeightsMatrix(new double[,] { { 1, 2, 3 }, { 2, 3, 4 } });

        NetworkVector vector_1 = new NetworkVector(new double[] { 1 });
        NetworkVector vector_2 = new NetworkVector(new double[] { 11, 12 });
        NetworkVector vector_3 = new NetworkVector(new double[] { 111, 112, 113 });


        public WeightedCombinerTests()
        {
            wc_1 = new WeightedCombiner(matrix_1);
            wc_1b = new WeightedCombiner(matrix_1, vector_1);
            wc_2 = new WeightedCombiner(matrix_2, vector_2);
        }

        [TestMethod]
        public void CanMake()
        {
            Assert.IsNotNull(wc_1);
            Assert.AreEqual(matrix_1, wc_1.Weights);
        }

        [TestMethod]
        public void CanMakeWithBiases()
        {
            Assert.IsNotNull(wc_1b);
            Assert.AreEqual(matrix_1, wc_1b.Weights);
            Assert.AreEqual(vector_1, wc_1b.Biases);
        }

        [TestMethod]
        public void CorrectDimensions1()
        {
            Assert.AreEqual(1, wc_1b.NumberOfInputs);
            Assert.AreEqual(wc_1b.NumberOfInputs, wc_1b.Input.Dimension);
            Assert.AreEqual(wc_1b.NumberOfInputs, wc_1b.Weights.NumberOfInputs);

            Assert.AreEqual(1, wc_1b.NumberOfOutputs);
            Assert.AreEqual(wc_1b.NumberOfOutputs, wc_1b.Output.Dimension);
            Assert.AreEqual(wc_1b.NumberOfOutputs, wc_1b.Weights.NumberOfOutputs);
            Assert.AreEqual(wc_1b.NumberOfOutputs, wc_1b. Biases.Dimension);
        }

        [TestMethod]
        public void CorrectDimensions2()
        {
            Assert.AreEqual(1, wc_1b.NumberOfInputs);
            Assert.AreEqual(wc_1b.NumberOfInputs, wc_1b.Input.Dimension);
            Assert.AreEqual(wc_1b.NumberOfInputs, wc_1b.Weights.NumberOfInputs);

            Assert.AreEqual(1, wc_1b.NumberOfOutputs);
            Assert.AreEqual(wc_1b.NumberOfOutputs, wc_1b.Output.Dimension);
            Assert.AreEqual(wc_1b.NumberOfOutputs, wc_1b.Weights.NumberOfOutputs);
            Assert.AreEqual(wc_1b.NumberOfOutputs, wc_1b.Biases.Dimension);
        }

        [TestMethod]
        public void CanRun1()
        {
            wc_1b.Run(vector_1);

            NetworkVector outcheck = new NetworkVector(new double[] { 2 });
            Assert.AreEqual(outcheck, wc_1b.Output);
        }

        [TestMethod]
        public void CanRun2()
        {
            wc_2.Run(vector_3);

            NetworkVector outcheck = new NetworkVector(new double[] { 111+224+339 + 11, 222+336+452 + 12});
            Assert.AreEqual(outcheck, wc_2.Output);
        }

        [TestMethod]
        public void CanBack1()
        {
            wc_1b.Run(vector_1);

            NetworkVector inputGradientCheck = vector_1.Copy();
            NetworkVector biasesGradientCheck = vector_1.Copy();
            WeightsMatrix weightsGradientCheck = matrix_1.Copy();

            Assert.AreEqual(inputGradientCheck, wc_1b.InputGradient(vector_1));
            Assert.AreEqual(biasesGradientCheck, wc_1b.BiasesGradient(vector_1));
            Assert.AreEqual(weightsGradientCheck, wc_1b.WeightsGradient(vector_1));
        }

        [TestMethod]
        public void CanBack2()
        {
            wc_2.Run(vector_3);

            NetworkVector inputGradientCheck = new NetworkVector(
                new double[] { 11*1 + 12*2, 11*2 + 12*3, 11*3 + 12*4  }
                );
            NetworkVector biasesGradientCheck = new NetworkVector(
                new double[] { 11, 12 }
                );
            WeightsMatrix weightsGradientCheck = new WeightsMatrix(
                new double[,] { { 11*111, 11*112, 11*113 }, { 12*111, 12*112, 12*113 } }
                );

            Assert.AreEqual(inputGradientCheck, wc_2.InputGradient(vector_2));
            Assert.AreEqual(biasesGradientCheck, wc_2.BiasesGradient(vector_2));
            Assert.AreEqual(weightsGradientCheck, wc_2.WeightsGradient(vector_2));
        }

        [TestMethod]
        public void CanUpdate()
        {
            wc_2.Run(vector_3);

            AdaptationStrategy strategy = new GradientDescent(1.0, 1);
            wc_2.BackPropagate(vector_2);
            wc_2.Update(strategy);
            
            NetworkVector biasesCheck = new NetworkVector(new double[] { 0, 0});
            WeightsMatrix weightsCheck = new WeightsMatrix(new double[,] { { 1 - (11 * 111), 2 - (11 * 112), 3 - (11 * 113) }, { 2 -  (12 * 111), 3 - (12 * 112), 4 - (12 * 113) } });
            Assert.AreEqual(biasesCheck, wc_2.Biases);
            Assert.AreEqual(weightsCheck, wc_2.Weights);

        }
    }
}
