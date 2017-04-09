using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;

using MathNet.Numerics.LinearAlgebra;
using NeuralNet2;

namespace TestNeuralNet
{
    [TestClass]
    public class DataVectorTests
    {
        [TestMethod]
        public void CanMake()
        {
            try
            {
                DataVector dv1 = new DataVector(Vector<double>.Build.Dense(1));
                DataVector dv2 = new DataVector(1);
            }
            catch (Exception e)
            {
                Assert.Fail("DataVector constructor threw and exception: " + e.Message);
            }
        }

        [TestMethod]
        public void CannotMakeBad()
        {
            try
            {
                DataVector dv1 = new DataVector(Vector<double>.Build.Dense(0));
                DataVector dv2 = new DataVector(0);
                Assert.Fail("DataVector constructor failed to throw an exception: ");
            }
            catch (ArgumentOutOfRangeException)
            { }
        }

        [TestMethod]
        public void CanAdd()
        {
            DataVector dv1 = new DataVector(Vector<double>.Build.DenseOfArray (new double[] { 1, 2 }));
            DataVector dv2 = new DataVector(Vector<double>.Build.DenseOfArray(new double[] { 2, 3 }));
            DataVector dv3 = dv1.Add(dv2);

            Assert.AreEqual(3, dv3[0]);
            Assert.AreEqual(5, dv3[1]);
        }

        [TestMethod]
        public void CannotAddMismatched()
        {
            try
            {
                DataVector dv1 = new DataVector(Vector<double>.Build.DenseOfArray(new double[] { 1, 2 }));
                DataVector dv2 = new DataVector(Vector<double>.Build.DenseOfArray(new double[] { 2, 3, 4 }));
                DataVector dv3 = dv1.Add(dv2);

                Assert.Fail("Add failed to throw an ArgumentOutOfRangeException.");
            }
            catch (ArgumentOutOfRangeException)
            { }
        }

        [TestMethod]
        public void CanSubtract()
        {
            DataVector dv1 = new DataVector(Vector<double>.Build.DenseOfArray(new double[] { 1, 2 }));
            DataVector dv2 = new DataVector(Vector<double>.Build.DenseOfArray(new double[] { 2, 3 }));
            DataVector dv3 = dv2.Subtract(dv1);

            Assert.AreEqual(1, dv3[0]);
            Assert.AreEqual(1, dv3[1]);
        }

        [TestMethod]
        public void CannotSubractMismatched()
        {
            try
            {
                DataVector dv1 = new DataVector(Vector<double>.Build.DenseOfArray(new double[] { 1, 2 }));
                DataVector dv2 = new DataVector(Vector<double>.Build.DenseOfArray(new double[] { 2, 3, 4 }));
                DataVector dv3 = dv2.Subtract(dv1);

                Assert.Fail("Subtract failed to throw an ArgumentOutOfRangeException.");
            }
            catch (ArgumentOutOfRangeException)
            { }
        }
    }
}
