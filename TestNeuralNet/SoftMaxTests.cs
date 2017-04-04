using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;

using NeuralNet;

namespace TestNeuralNet
{
    [TestClass]
    public class SoftMaxTests
    {
        //[TestMethod]
        //public void CanMakeSoftMax()
        //{
        //    Assert.Fail("Need to adapt soft-max unit.");
        //}

        [TestMethod]
        public void CanMakeSoftMax()
        {
            //SoftMaxUnit smu = new SoftMaxUnit(1);
            SoftMaxUnit smu = new SoftMaxUnit();  // just a flag, so we do not forget to adapt
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

        //[TestMethod]
        //public void UnrunSoftMaxHasZeroOutput()
        //{
        //    SoftMaxUnit smu = new SoftMaxUnit(7);
        //    double[] outputvalues = smu.Output.ToArray();
        //    for (int i = 0; i < smu.NumberOfOutputs; i++)
        //    {
        //        Assert.AreEqual(0, outputvalues[i]);
        //    }
        //}

        [TestMethod]
        public void CannotRunWithWrongSizedInput()
        {
            SoftMaxUnit smu = new SoftMaxUnit(7);
            NetworkVector badinput = new NetworkVector(new double[] { 1, 2, 3 });
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
            NetworkVector input = new NetworkVector(new double[] { 1, 0, 0, 0, 0, 0, 1 });

            NetworkVector output = smu.Run(input);

            double sum = 5 + (2 * Math.E);
            double one_value = Math.E / sum;
            double zero_value = 1 / sum;
            double delta = 0.000000001;

            double[] outputvalues = output.ToArray();
            double outputvaluessum = output.SumValues();
            Assert.AreEqual(1.0, outputvaluessum, delta);
            Assert.AreEqual(one_value, outputvalues[0], delta);
            Assert.AreEqual(one_value, outputvalues[6], delta);
            for (int i = 1; i < smu.NumberOfOutputs - 1; i++)
            {
                Assert.AreEqual(zero_value, outputvalues[i], delta);
            }
        }


        [TestMethod]
        public void CannotRunInputGradientWithWrongSizedOutputGradient()
        {
            SoftMaxUnit smu = new SoftMaxUnit(3);
            NetworkVector goodinput = new NetworkVector(new double[] { 1, 2, 3 });
            NetworkVector badgradient = new NetworkVector(new double[] { 1 });
            smu.Run(goodinput);
            try
            {
                smu.InputGradient(badgradient);
                Assert.Fail("Run failed to throw an ArgumentException when the input size did not match the number of units.");
            }
            catch (ArgumentException) { }
        }

        [TestMethod]
        public void InputGradientIsCorrect()
        {
            SoftMaxUnit smu = new SoftMaxUnit(7);
            NetworkVector inputs = new NetworkVector(new double[] { 0, 1, 0, 0, 1, 0, 0 });
            NetworkVector outputgradient = new NetworkVector(new double[] { 1, 1, 0, 0, 0, 0, 0 });

            NetworkVector output = smu.Run(inputs);
            NetworkVector inputGradient =  smu.InputGradient(outputgradient, output);

            double sum = 5 + (2 * Math.E);
            double one_value = Math.E / sum;
            double one_derivative = one_value * (1 - one_value);
            double zero_value = 1 / sum;
            double zero_derivative = zero_value * (1 - zero_value);

            NetworkVector inputGradientCheck = new NetworkVector(new double[]
            {
                zero_derivative,
                one_derivative,
                0, 0, 0, 0 , 0
            }
            );
            Assert.AreEqual(inputGradientCheck, inputGradient);
            
        }
    }
}
