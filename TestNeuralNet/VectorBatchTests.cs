using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;

using NeuralNet;
using MathNet.Numerics.LinearAlgebra;
using System.Collections.Generic;

namespace TestNeuralNet
{
    [TestClass]
    public class VectorBatchTests
    {
        [TestMethod]
        public void CanMake_IEnumerable()
        {
            Vector<double> vector1 = Vector<double>.Build.Dense(new double[] { 0, 1, 2, 3 });
            Vector<double> vector2 = Vector<double>.Build.Dense(new double[] { 1, 2, 3, 4 });
            List<NetworkVector> vectors = 
                new List<NetworkVector> { new NetworkVector( vector1), new NetworkVector( vector2 ) };

            try
            {
                VectorBatch batch = new VectorBatch(vectors);
            }
            catch (Exception e)
            {
                Assert.Fail("VectorBatch constructor threw an exception. Message: " + e.Message);
            }
        }

        [TestMethod]
        public void CanMake_Matrix()
        {
            Vector<double> vector1 = Vector<double>.Build.Dense(new double[] { 0, 1, 2, 3 });
            Vector<double> vector2 = Vector<double>.Build.Dense(new double[] { 1, 2, 3, 4 });
            Matrix<double> vectors =
                Matrix<double>.Build.DenseOfRowVectors(
                    new List<Vector<double>> { vector1, vector2 });

            try
            {
                VectorBatch batch = new VectorBatch(vectors);
            }
            catch (Exception e)
            {
                Assert.Fail("VectorBatch constructor threw an exception. Message: " + e.Message);
            }
        }

        [TestMethod]
        public void Dimension()
        {
            Vector<double> vector1 = Vector<double>.Build.Dense(new double[] { 0, 1, 2, 3 });
            Vector<double> vector2 = Vector<double>.Build.Dense(new double[] { 1, 2, 3, 4 });
            List<NetworkVector> vectors =
                new List<NetworkVector> { new NetworkVector(vector1), new NetworkVector(vector2) };
            
            VectorBatch batch = new VectorBatch(vectors);
            Assert.AreEqual(vectors[0].Dimension, batch.Dimension);
        }
    }
}
