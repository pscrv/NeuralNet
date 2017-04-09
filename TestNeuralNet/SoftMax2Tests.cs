using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;

using NeuralNet2;
using MathNet.Numerics.LinearAlgebra;

namespace TestNeuralNet
{

    [TestClass]
    public class SoftMax2Tests
    {

        [TestMethod]
        public void CanMake()
        {
            try
            {
                SoftMaxUnit smu = new SoftMaxUnit(1);
            }
            catch (Exception e)
            {
                var x = e.GetType();
            }
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
        public void CannotRunWithWrongSizedInput()
        {
            SoftMaxUnit smu = new SoftMaxUnit(7);
            DataVector badinput = new DataVector(Vector<double>.Build.DenseOfArray( new double[] { 1, 2, 3 }) );
            try
            {
                smu.Run(badinput);
                Assert.Fail("Run failed to throw an ArgumentException when the input size did not match the number of units.");
            }
            catch (ArgumentException) { }
        }


        [TestMethod]
        public void CorrectRun()
        {
            SoftMaxUnit smu = new SoftMaxUnit(4);
            DataVector input = new DataVector(
                Vector<double>.Build.DenseOfArray( new double[] { 1, 0, 0, 1 }) 
                );

            DataVector result = smu.Run(input);

            double sum = 2 + (2 * Math.Exp(-1));
            double zero_value = Math.Exp(-1) / sum;
            double one_value = 1 / sum;

            DataVector check = new DataVector(
                Vector<double>.Build.DenseOfArray( new double[] { one_value, zero_value, zero_value, one_value})
                );

            Assert.AreEqual(check, result);
        }


        [TestMethod]
        public void CannotBackPropagatetWithWrongSizedOutputGradient()
        {
            SoftMaxUnit smu = new SoftMaxUnit(3);
            DataVector goodinput = new DataVector(
                Vector<double>.Build.DenseOfArray( new double[] { 1, 2, 3 }) );
            DataVector badgradient = new DataVector(
                Vector<double>.Build.DenseOfArray(new double[] { 1 }) );
            smu.Run(goodinput);
            try
            {
                smu.BackPropagate(badgradient);
                Assert.Fail("Run failed to throw an ArgumentException when the input size did not match the number of units.");
            }
            catch (ArgumentException) { }
        }

        [TestMethod]
        public void InputGradientIsCorrect()
        {
            SoftMaxUnit smu = new SoftMaxUnit(7);
            DataVector inputs = new DataVector(
                Vector<double>.Build.DenseOfArray(new double[] { 0, 1, 0, 0, 1, 0, 0 }) );
            DataVector outputgradient = new DataVector(
                Vector<double>.Build.DenseOfArray(new double[] { 1, 1, 0, 0, 0, 0, 0 }) );

            DataVector output = smu.Run(inputs);
            DataVector inputGradient = smu.BackPropagate(outputgradient);

            double sum = 2 + (5 * Math.Exp(-1));
            double zero_value = Math.Exp(-1) / sum;
            double one_value = 1 / sum;
            double one_derivative = one_value * (1 - one_value);
            double zero_derivative = zero_value * (1 - zero_value);

            DataVector inputGradientCheck = new DataVector(
                Vector<double>.Build.DenseOfArray(new double[]
                {
                    zero_derivative,
                    one_derivative,
                    0, 0, 0, 0 , 0
                }
                ));

            Assert.AreEqual(inputGradientCheck, inputGradient);
        }
    }
}
