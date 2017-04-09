using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;

using MathNet.Numerics.LinearAlgebra;
using NeuralNet2;

namespace TestNeuralNet
{
    [TestClass]
    public class VectorBatch2Tests
    {
        [TestMethod]
        public void CanMake()
        {
            try
            {
                VectorBatch vb1 = new VectorBatch(Matrix<double>.Build.Dense(1, 1));
                VectorBatch vb2 = new VectorBatch(
                    Matrix<double>.Build.DenseOfArray(
                        new double[,] { { 1, 2 }, { 2, 3 } }
                        )
                    );
                VectorBatch vb3 = new VectorBatch(vb1);
            }
            catch (Exception e)
            {
                Assert.Fail("VectorBatch constructor threw an exception: " + e.Message);
            }
        }




        [TestMethod]
        public void CanAdd()
        {
            VectorBatch vb1 = new VectorBatch(Matrix<double>.Build.DenseOfArray(new double[,] { { 1, 2, 3 }, { 2, 3, 4} }));
            VectorBatch vb2 = new VectorBatch(Matrix<double>.Build.DenseOfArray(new double[,] { { 2, 3, 4 }, { 3, 4, 5} }));
            VectorBatch vb3 = vb1.Add(vb2);

            Assert.AreEqual(3, vb3[0, 0]);
            Assert.AreEqual(5, vb3[0, 1]);
            Assert.AreEqual(7, vb3[0, 2]);
            Assert.AreEqual(5, vb3[1, 0]);
            Assert.AreEqual(7, vb3[1, 1]);
            Assert.AreEqual(9, vb3[1, 2]);
        }

        [TestMethod]
        public void CannotAddMismatched()
        {
            try
            {
                VectorBatch vb1 = new VectorBatch(Matrix<double>.Build.DenseOfArray(new double[,] { { 1, 2 }, { 2, 3 } }));
                VectorBatch vb2 = new VectorBatch(Matrix<double>.Build.DenseOfArray(new double[,] { { 2, 3, 4 }, { 3, 4, 5 } }));
                VectorBatch vb3 = vb1.Add(vb2);

                Assert.Fail("Add failed to throw an ArgumentOutOfRangeException.");
            }
            catch (ArgumentOutOfRangeException)
            { }
        }

        [TestMethod]
        public void CanSubtract()
        {
            VectorBatch vb1 = new VectorBatch(Matrix<double>.Build.DenseOfArray(new double[,] { { 1, 2, 3 }, { 2, 3, 4 } }));
            VectorBatch vb2 = new VectorBatch(Matrix<double>.Build.DenseOfArray(new double[,] { { 2, 3, 4 }, { 3, 4, 5 } }));
            VectorBatch vb3 = vb2.Subtract(vb1);

            Assert.AreEqual(1, vb3[0, 0]);
            Assert.AreEqual(1, vb3[0, 1]);
            Assert.AreEqual(1, vb3[0, 2]);
            Assert.AreEqual(1, vb3[1, 0]);
            Assert.AreEqual(1, vb3[1, 1]);
            Assert.AreEqual(1, vb3[1, 2]);
        }

        [TestMethod]
        public void CannotSubractMismatched()
        {
            try
            {
                VectorBatch vb1 = new VectorBatch(Matrix<double>.Build.DenseOfArray(new double[,] { { 1, 2 }, { 2, 3 } }));
                VectorBatch vb2 = new VectorBatch(Matrix<double>.Build.DenseOfArray(new double[,] { { 2, 3, 4 }, { 3, 4, 5 } }));
                VectorBatch vb3 = vb2.Subtract(vb1);

                Assert.Fail("Subtract failed to throw an ArgumentOutOfRangeException.");
            }
            catch (ArgumentOutOfRangeException)
            { }
        }

    }



}
