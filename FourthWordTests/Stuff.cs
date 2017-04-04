using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;

using NeuralNet;
using MathNet.Numerics.LinearAlgebra;
using System.Collections.Generic;

namespace FourthWordTests
{
    [TestClass]
    public class Stuff
    {

        [TestMethod]
        public void AsMatrix()
        {
            NetworkVector vector = new NetworkVector(new double[] { 1, 2, 3 });
            VectorBatch batch = new VectorBatch(new List<NetworkVector> { vector, vector });
            Matrix<double> result = batch.AsMatrix();
        }
    }
}
