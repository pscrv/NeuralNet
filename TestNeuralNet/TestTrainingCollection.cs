using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;

using NeuralNet;

namespace TestNeuralNet
{
    [TestClass]
    public class TestTrainingCollection
    {
        [TestMethod]
        public void CanMakeAndAdd()
        {
            TrainingCollection tc = new TrainingCollection();
            tc.Add(new VectorPair(
                new NetworkVector(1),
                new NetworkVector(2)
                ));
            Assert.IsNotNull(tc);
            Assert.AreEqual(1, tc.Count);
        }

        [TestMethod]
        public void CanIterate()
        {
            TrainingCollection tc = new TrainingCollection();
            tc.Add(new VectorPair(
                new NetworkVector(1),
                new NetworkVector(2)
                ));
            tc.Add(new VectorPair(
                new NetworkVector(1),
                new NetworkVector(2)
                ));
            int count = 0;
            foreach (VectorPair pair in tc)
            {
                count++;
            }
            Assert.IsNotNull(tc);
            Assert.AreEqual(count, tc.Count);
        }

        [TestMethod]
        public void CanMakeGetAsSingletons()
        {
            TrainingCollection tc = new TrainingCollection();
            for (int i = 0; i < 10; i++)
            {
                tc.Add(new VectorPair(
                    new NetworkVector(1),
                    new NetworkVector(2)
                    ));

            }
            Assert.IsNotNull(tc);
            Assert.AreEqual(10, tc.Count);

            int count = 0;
            foreach (TrainingCollection singleton in tc.AsSingletons())
            {
                Assert.AreEqual(1, singleton.Count);
                count++;
            }
            Assert.AreEqual(10, count);
        }

        [TestMethod]
        public void CanMakeGetAsBatches()
        {
            TrainingCollection tc = new TrainingCollection();
            for (int i = 0; i < 100; i++)
            {
                tc.Add(new VectorPair(
                    new NetworkVector(1),
                    new NetworkVector(2)
                    ));
            }
            Assert.IsNotNull(tc);
            Assert.AreEqual(100, tc.Count);

            int count = 0;
            foreach (TrainingCollection singleton in tc.AsBatches(10))
            {
                Assert.AreEqual(10, singleton.Count);
                count++;
            }
            Assert.AreEqual(10, count);
        }
    }
}
