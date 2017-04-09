using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;

using MathNet.Numerics.LinearAlgebra;
using NeuralNet2;

namespace TestNeuralNet
{
    [TestClass]
    public class BiasesVectorTests
    {
        [TestMethod]
        public void CanMake()
        {
            try
            {
                BiasesVector bv1 = new BiasesVector(Vector<double>.Build.Dense(1));
                BiasesVector bv2 = new BiasesVector(1);
            }
            catch (Exception e)
            {
                Assert.Fail("BiasesVector constructor threw and exception: " + e.Message);
            }
        }

        [TestMethod]
        public void CannotMakeBad()
        {
            try
            {
                BiasesVector bv1 = new BiasesVector(Vector<double>.Build.Dense(0));
                BiasesVector bv2 = new BiasesVector(0);
                Assert.Fail("BiasesVector constructor failed to throw an exception: ");
            }
            catch (ArgumentOutOfRangeException)
            { }
        }

        [TestMethod]
        public void CanAdd()
        {
            BiasesVector bv1 = new BiasesVector(Vector<double>.Build.DenseOfArray(new double[] { 1, 2 }));
            BiasesVector bv2 = new BiasesVector(Vector<double>.Build.DenseOfArray(new double[] { 2, 3 }));
            BiasesVector bv3 = bv1.Add(bv2);

            Assert.AreEqual(3, bv3[0]);
            Assert.AreEqual(5, bv3[1]);
        }

        [TestMethod]
        public void CannotAddMismatched()
        {
            try
            {
                BiasesVector bv1 = new BiasesVector(Vector<double>.Build.DenseOfArray(new double[] { 1, 2 }));
                BiasesVector bv2 = new BiasesVector(Vector<double>.Build.DenseOfArray(new double[] { 2, 3, 4 }));
                BiasesVector bv3 = bv1.Add(bv2);

                Assert.Fail("Add failed to throw an ArgumentOutOfRangeException.");
            }
            catch (ArgumentOutOfRangeException)
            { }
        }

        [TestMethod]
        public void CanSubtract()
        {
            BiasesVector bv1 = new BiasesVector(Vector<double>.Build.DenseOfArray(new double[] { 1, 2 }));
            BiasesVector bv2 = new BiasesVector(Vector<double>.Build.DenseOfArray(new double[] { 2, 3 }));
            BiasesVector bv3 = bv2.Subtract(bv1);

            Assert.AreEqual(1, bv3[0]);
            Assert.AreEqual(1, bv3[1]);
        }

        [TestMethod]
        public void CannotSubractMismatched()
        {
            try
            {
                BiasesVector bv1 = new BiasesVector(Vector<double>.Build.DenseOfArray(new double[] { 1, 2 }));
                BiasesVector bv2 = new BiasesVector(Vector<double>.Build.DenseOfArray(new double[] { 2, 3, 4 }));
                BiasesVector bv3 = bv2.Subtract(bv1);

                Assert.Fail("Subtract failed to throw an ArgumentOutOfRangeException.");
            }
            catch (ArgumentOutOfRangeException)
            { }
        }
    }
}
