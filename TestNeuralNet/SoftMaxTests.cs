using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;

using NeuralNet;

namespace TestNeuralNet
{
    [TestClass]
    public class SoftMaxTests
    {
        [TestMethod]
        public void CanMakeSoftMax()
        {
            SoftMaxUnit smu = new SoftMaxUnit(1);
            Assert.AreNotEqual(null, smu);
        }

        [TestMethod]
        public void SoftMaxHasCorrectNumberOfInputsAndOutputs()
        {
            int testvalue = 3;
            SoftMaxUnit smu = new SoftMaxUnit(testvalue);
            Assert.AreEqual(testvalue, smu.NumberOfInputs);
            Assert.AreEqual(testvalue, smu.NumberOfOutputs);
        }

        [TestMethod]
        public void UnrunSoftMaxHasZeroOutput()
        {
            SoftMaxUnit smu = new SoftMaxUnit(7);
            double[] outputvalues = smu.Output.ToArray();
            for (int i = 0; i < smu.NumberOfOutputs; i++)
            {
                Assert.AreEqual(0, outputvalues[i]);
            }
        }

        [TestMethod]
        public void CannotRunWithWrongSizedInput()
        {
            SoftMaxUnit smu = new SoftMaxUnit(7);
            NetworkVector badinput = new NetworkVector( new double[] { 1, 2, 3} );
            try
            {
                smu.Run(badinput);
                Assert.Fail("Run failed to throw an ArgumentException when the input size did not match the number of units.");
            }
            catch (ArgumentException) { }
        }


        [TestMethod]
        public void RunProducesCorrectOutput()
        {
            SoftMaxUnit smu = new SoftMaxUnit(7);
            NetworkVector inputs = new NetworkVector( new double[] { 1, 0, 0, 0, 0, 0, 1 });

            smu.Run(inputs);

            double sum = 5 + (2 * Math.E);
            double one_value = Math.E / sum;
            double zero_value = 1 / sum;
            double delta = 0.000000001;

            double[] outputvalues = smu.Output.ToArray();
            double outputvaluessum = smu.Output.SumValues();
            Assert.AreEqual(1.0, outputvaluessum, delta);
            Assert.AreEqual(one_value, outputvalues[0], delta);
            Assert.AreEqual(one_value, outputvalues[6], delta);
            for (int i = 1; i < smu.NumberOfOutputs - 1; i++)
            {
                Assert.AreEqual(zero_value, outputvalues[i], delta);
            }
        }


        [TestMethod]
        public void CannotBackPropagateWithWrongSizedOutputGradient()
        {
            SoftMaxUnit smu = new SoftMaxUnit(3);
            NetworkVector goodinput = new NetworkVector(new double[] { 1, 2, 3 });
            NetworkVector badgradient = new NetworkVector(new double[] { 1 });
            smu.Run(goodinput);
            try
            {
                smu.BackPropagate(badgradient);
                Assert.Fail("Run failed to throw an ArgumentException when the input size did not match the number of units.");
            }
            catch (ArgumentException) { }
        }

        [TestMethod]
        public void BackPropagationProducesCorrectInputGradient()
        {
            SoftMaxUnit smu = new SoftMaxUnit(7);
            NetworkVector inputs = new NetworkVector(new double[] { 0, 1, 0, 0, 1, 0, 0 });
            NetworkVector outputgradient = new NetworkVector(new double[] { 1, 1, 0, 0, 0, 0, 0 });

            smu.Run(inputs);
            smu.BackPropagate(outputgradient);

            double sum = 5 + (2 * Math.E);
            double one_value = Math.E / sum;
            double one_derivative = one_value * (1 - one_value);
            double zero_value = 1 / sum;
            double zero_derivative = zero_value * (1 - zero_value);
            double delta = 0.000000001;

            double[] inputgradientvalues = smu.InputGradient.ToArray();
            Assert.AreEqual(zero_derivative, inputgradientvalues[0], delta);
            Assert.AreEqual(one_derivative, inputgradientvalues[1], delta);
            for (int i = 2; i < smu.NumberOfOutputs; i++)
            {
                Assert.AreEqual(0, inputgradientvalues[i], delta);
            }
        }
    }
}
