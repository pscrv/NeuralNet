using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;

using NeuralNet;

namespace TestNeuralNet
{
    [TestClass]
    public class GradientDescentTests
    {
        [TestMethod]
        public void CanMake()
        {
            try
            {
                GradientDescent gd = new GradientDescent();
            }
            catch (Exception e)
            {
                Assert.Fail(string.Format("GradientDescent constructor threw exception: {0}", e));
            }
        }

        [TestMethod]
        public void CanMakeWithStepSize()
        {
            try
            {
                GradientDescent gd = new GradientDescent(0.5);
            }
            catch (Exception e)
            {
                Assert.Fail(string.Format("GradientDescent constructor threw exception: {0}", e));
            }
        }

        [TestMethod]
        public void CanGetBiasesUpdate()
        {
            GradientDescent gd = new GradientDescent(0.5);
            NetworkVector testVector = new NetworkVector(new double[] { 1, 2, 3 });
            NetworkVector result = gd.BiasesUpdate(testVector);
            NetworkVector resultCheck = new NetworkVector(new double[] { -0.5, -1.0, -1.5 });

            Assert.AreEqual(resultCheck, result);
        }

        [TestMethod]
        public void CanGetWeightsUpdate()
        {
            GradientDescent gd = new GradientDescent(0.5);
            NetworkMatrix testMatrix = new NetworkMatrix(new double[,] { { 1, 2, 3 }, { 2, 3, 4 } });
            NetworkMatrix result = gd.WeightsUpdate(testMatrix);
            NetworkMatrix resultCheck = new NetworkMatrix(new double[,] { { -0.5, -1.0, -1.5 }, { -1.0, -1.5, -2.0 } });

            Assert.AreEqual(resultCheck, result);
        }
    }
}
