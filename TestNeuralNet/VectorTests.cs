using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NeuralNet;
using System.Collections.Generic;

namespace TestNeuralNet
{
    [TestClass]
    public class VectorTests
    {
        [TestMethod]
        public void CanMakeFullVector()
        {
            Vector vector;
            try
            {
                vector = new FullVector(new double[] { 1 });
                vector = new FullVector(1);
                vector = new FullVector(vector);
            }
            catch (Exception e)
            {
                Assert.Fail("FullVector() threw an exception with message: " + e.Message);
            }
        }

        [TestMethod]
        public void CanMakeUnitVector()
        {
            Vector vector;
            try
            {
                vector = new UnitVector(0, 1);
            }
            catch (Exception e)
            {
                Assert.Fail("UnitVector() threw an exception with message: " + e.Message);
            }
        }

        [TestMethod]
        public void CanMakeCompositeVector()
        {
            try
            {
                Vector vector1 = new FullVector(new double[] { 1 });
                Vector vector2 = new FullVector(new double[] { 2, 3 });
                List<Vector> vectorList = new List<Vector> { vector1, vector2 };
                Vector vector = new CompositeVector(vectorList);
            }
            catch (Exception e)
            {
                Assert.Fail("FullVector() threw an exception with message: " + e.Message);
            }
        }


        [TestMethod]
        public void AsVector()
        {
            Vector vector1 = new FullVector(new double[] { 1 });
            Vector vector2 = new UnitVector(0, 1);
            Vector vector3 = new CompositeVector(new List<Vector> { vector1, vector2 });

            Vector result = vector1.AsFullVector();
            Assert.AreEqual(result, vector1);
            Assert.IsInstanceOfType(result, typeof(FullVector));

            result = vector2.AsFullVector();
            Assert.AreEqual(result, vector2);
            Assert.IsInstanceOfType(result, typeof(FullVector));

            result = vector3.AsFullVector();
            Assert.AreEqual(result, vector3);
            Assert.IsInstanceOfType(result, typeof(FullVector));
            Assert.AreEqual(2, result.Length);

        }

    }
}
