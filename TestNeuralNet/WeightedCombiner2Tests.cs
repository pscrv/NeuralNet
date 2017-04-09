using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;

using MathNet.Numerics.LinearAlgebra;
using NeuralNet2;

namespace TestNeuralNet
{

    [TestClass]
    public class WeightedCombiner2Tests
    {
        WeightedCombiner wc_1;
        WeightedCombiner wc_1b;
        WeightedCombiner wc_2;

        WeightsMatrix matrix_1 = new WeightsMatrix(
            Matrix<double>.Build.DenseOfArray( new double[,] { { 1 } })
            );

        WeightsMatrix matrix_2 = new WeightsMatrix(
            Matrix<double>.Build.DenseOfArray( new double[,] { { 1, 2, 3 }, { 2, 3, 4 } })
            );

        BiasesVector biases_1 = new BiasesVector(
            Vector<double>.Build.DenseOfArray( new double[] {  1  }) 
            );

        BiasesVector biases_2 = new BiasesVector(
                Vector<double>.Build.DenseOfArray( new double[] { 11, 12 } ) 
            );

        VectorBatch inputBatch_1 = new VectorBatch(
            Matrix<double>.Build.DenseOfArray( new double[,] { { 111 }, { 112 }, { 113 } }) 
            );

        VectorBatch inputBatch_2 = new VectorBatch(
            Matrix<double>.Build.DenseOfArray(new double[,] { { 1, 2, 3 }, { 2, 3, 4 } }) 
            );

        VectorBatch gradientBatch1 = new VectorBatch(
            Matrix<double>.Build.DenseOfArray(new double[,] { { 2 }, { 2 }, { 2 } })
            );

        VectorBatch gradientBatch2 = new VectorBatch(
            Matrix<double>.Build.DenseOfArray(new double[,] { { 1, 2 }, { 2, 3 } })
            );

        VectorBatch inputGradientCheck = new VectorBatch(
            Matrix<double>.Build.DenseOfArray(new double[,] { { 5, 8, 11 }, { 8, 13, 18 } })
            );

        public WeightedCombiner2Tests()
        {
            try
            {
                wc_1 = new WeightedCombiner(matrix_1);
                wc_1b = new WeightedCombiner(matrix_1, biases_1);
                wc_2 = new WeightedCombiner(matrix_2, biases_2);
            }
            catch (Exception e)
            {
                Assert.Fail("WeigthedCombiner constructor threw an exception: " + e.Message);
            }
        }



        [TestMethod]
        public void CanMake()
        {
            Assert.IsNotNull(wc_1);
        }

        [TestMethod]
        public void CanMakeWithBiases()
        {
            Assert.IsNotNull(wc_1b);
        }

        [TestMethod]
        public void CanRun1()
        {
            VectorBatch result = wc_1.Run(inputBatch_1);
            VectorBatch outcheck = inputBatch_1;
            Assert.AreEqual(outcheck, result);
        }

        [TestMethod]
        public void CanRun2()
        {
            VectorBatch result = wc_2.Run(inputBatch_2);

            VectorBatch outcheck = new VectorBatch(
                Matrix<double>.Build.DenseOfArray( new double[,] { { 25, 32 }, { 31, 41 } }) );
            Assert.AreEqual(outcheck, result);
        }


        [TestMethod]
        public void CanBack1()
        {
            wc_1b.Run(inputBatch_1);
            VectorBatch inputGradient = wc_1b.BackPropagate(gradientBatch1);

            VectorBatch inputGradientCheck = gradientBatch1;

            Assert.AreEqual(inputGradientCheck, inputGradient);
        }

        [TestMethod]
        public void CanBack2()
        {
            wc_2.Run(inputBatch_2);
            VectorBatch inputGradient = wc_2.BackPropagate(gradientBatch2);

            Assert.AreEqual(inputGradientCheck, inputGradient);
        }


        [TestMethod]
        public void RunAfterBackPropIsCorrect()
        {
            VectorBatch result;
            result = wc_2.Run(inputBatch_2);
            VectorBatch inputGradient = wc_2.BackPropagate(gradientBatch2);
            result = wc_2.Run(inputBatch_2);

            VectorBatch resultCheck = new VectorBatch(
                Matrix<double>.Build.DenseOfArray(new double[,] { { -32, -61 }, { -50, -91 } })
                );

            Assert.AreEqual(resultCheck, result);
        }


    }
    
}
