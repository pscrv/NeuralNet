using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;

using NeuralNet;

namespace TestNeuralNet
{
    [TestClass]
    public class WeightedCombinerBankTests
    {
        [TestMethod]
        public void CanMake()
        {
            WeightedCombiner combiner = new WeightedCombiner(new NetworkMatrix( new double[,] { { 1 } }) );
            WeightedCombinerBank bank = new WeightedCombinerBank(combiner, 1);
            Assert.AreNotEqual(null, bank);
        }

        [TestMethod]
        public void CannotMakeNullWeightedCombinerBank()
        {
            WeightedCombiner combiner = null;

            try
            {
                WeightedCombinerBank bank = new WeightedCombinerBank(combiner, 1);
                Assert.Fail("Failure to throw a NullReferenceException when creating a WeightedCombinerBank with a null layer.");
            }
            catch (NullReferenceException) { }
        }

        [TestMethod]
        public void CannotMakeZeroWeightedCombinerBank()
        {
            WeightedCombiner layer = new WeightedCombiner(new NetworkMatrix( new double[,] { { 1 } }));

            try
            {
                WeightedCombinerBank bank = new WeightedCombinerBank(layer, 0);
                Assert.Fail("Failure to throw an ArgumentException when creating a WeightedCombinerBank with a < 1 banks.");
            }
            catch (ArgumentException) { }
        }


        [TestMethod]
        public void WeightedCombinerBankHasCorrectOutputSize()
        {
            WeightedCombiner combiner1 = new WeightedCombiner(new NetworkMatrix( new double[,] { { 1 } }));
            WeightedCombiner combiner2 = new WeightedCombiner(new NetworkMatrix( new double[,] { { 1 }, { 2 }, { 3 } }));

            WeightedCombinerBank bank1a = new WeightedCombinerBank(combiner1, 1);
            WeightedCombinerBank bank1b = new WeightedCombinerBank(combiner1, 3);
            WeightedCombinerBank bank2a = new WeightedCombinerBank(combiner2, 2);
            WeightedCombinerBank bank2b = new WeightedCombinerBank(combiner2, 5);

            Assert.AreEqual(1, bank1a.NumberOfOutputs);
            Assert.AreEqual(bank1a.NumberOfOutputs, bank1a.Output.Dimension);

            Assert.AreEqual(3, bank1b.NumberOfOutputs);
            Assert.AreEqual(bank1b.NumberOfOutputs, bank1b.Output.Dimension);

            Assert.AreEqual(6, bank2a.NumberOfOutputs);
            Assert.AreEqual(bank2a.NumberOfOutputs, bank2a.Output.Dimension);

            Assert.AreEqual(15, bank2b.NumberOfOutputs);
            Assert.AreEqual(bank2b.NumberOfOutputs, bank2b.Output.Dimension);
        }

        [TestMethod]
        public void UnRunWeightedCombinerBankHasZeroOutput()
        {
            int numberOfBanks = 3;
            WeightedCombiner layer = new WeightedCombiner(new NetworkMatrix( new double[,] { { 1 } }));
            WeightedCombinerBank bank = new WeightedCombinerBank(layer, numberOfBanks);
            NetworkVector outputcheck = new NetworkVector(numberOfBanks);
            Assert.AreEqual(outputcheck, bank.Output);
        }

        [TestMethod]
        public void CanRunWeightedCombinerBank()
        {
            int numberOfBanks = 3;
            WeightedCombiner layer = new WeightedCombiner(new NetworkMatrix( new double[,] { { 1 } }));
            WeightedCombinerBank bank = new WeightedCombinerBank(layer, numberOfBanks);
            NetworkVector zeroVector = new NetworkVector(numberOfBanks);

            bank.Run(zeroVector);
            Assert.AreEqual(zeroVector, bank.Output);
        }


        [TestMethod]
        public void CannotRunWeightedCombinerBankWithBadInputSize()
        {
            int numberOfBanks = 3;
            WeightedCombiner layer = new WeightedCombiner(new NetworkMatrix( new double[,] { { 1 } }) );
            WeightedCombinerBank bank = new WeightedCombinerBank(layer, numberOfBanks);
            NetworkVector badInput = new NetworkVector(2);

            try
            {
                bank.Run(badInput);
                Assert.Fail("WeightedCombinerBank.Run failed to throw an ArgumentException for input of the wrong size.");
            }
            catch (ArgumentException) { }
        }

        [TestMethod]
        public void CorrectRunOneInput()
        {
            int numberOfBanks = 3;
            WeightedCombiner layer = new WeightedCombiner(new NetworkMatrix( new double[,] { { 1 } }) );
            WeightedCombinerBank bank = new WeightedCombinerBank(layer, numberOfBanks);
            NetworkVector input = new NetworkVector(new double[] { 2, 3, 5 });
            NetworkVector outputcheck = new NetworkVector(new double[] { 2, 3, 5 });

            bank.Run(input);
            Assert.AreEqual(outputcheck, bank.Output);
        }

        [TestMethod]
        public void CorrectRunThreeInputs()
        {
            int numberOfBanks = 3;
            WeightedCombiner layer = new WeightedCombiner(new NetworkMatrix( new double[,] { { 2, 3, 5 } }) );
            WeightedCombinerBank bank = new WeightedCombinerBank(layer, numberOfBanks);
            NetworkVector input = new NetworkVector(new double[] { 7, 7, 7, 11, 11, 11, 13, 13, 13 });
            NetworkVector outputcheck = new NetworkVector(new double[] { 70, 110, 130 });

            bank.Run(input);
            Assert.AreEqual(outputcheck, bank.Output);
        }

        [TestMethod]
        public void InputGradientWeightedCombinerBank()
        {
            int numberOfBanks = 3;
            WeightedCombiner layer = new WeightedCombiner(new NetworkMatrix( new double[,] { { 1 } }) );
            WeightedCombinerBank bank = new WeightedCombinerBank(layer, numberOfBanks);
            NetworkVector zeroVector = new NetworkVector(numberOfBanks);

            bank.BackPropagate(zeroVector);
            Assert.AreEqual(zeroVector, bank.InputGradient(zeroVector));
        }

        [TestMethod]
        public void BackPropagateWeightedCombinerBankIsCorrect2()
        {
            int numberOfBanks = 3;
            WeightedCombiner layer = new WeightedCombiner(new NetworkMatrix( new double[,] { { 1, 2 } } ), new NetworkVector( new double[] { 1 }) );
            WeightedCombinerBank bank = new WeightedCombinerBank(layer, numberOfBanks);
            NetworkVector inputVector = new NetworkVector(new double[] { 1, 1, 2, 2, 3, 3 });
            NetworkVector bpVector = new NetworkVector(new double[] { 5, 7, 11 });

            bank.Run(inputVector);

            NetworkVector inputGradientCheck = new NetworkVector(new double[] { 5, 10, 7, 14, 11, 22 });
            NetworkMatrix weightsCheck = new NetworkMatrix( new double[,] { { -51, -50 } } );
            NetworkVector biasCheck = new NetworkVector( new double[] { -22 } );
            
            Assert.AreEqual(inputGradientCheck, bank.InputGradient(bpVector));

            bank.BackPropagate(bpVector);
            bank.Update(new GradientDescent());
            Assert.AreEqual(biasCheck, bank.Biases);
            Assert.AreEqual(weightsCheck, bank.Weights);
        }

        [TestMethod]
        public void BackPropagateWeightedCombinerBankIsCorrect()
        {
            int numberOfBanks = 3;
            WeightedCombiner layer = new WeightedCombiner(new NetworkMatrix( new double[,] { { 1 } }) );
            WeightedCombinerBank bank = new WeightedCombinerBank(layer, numberOfBanks);
            NetworkVector inputVector = new NetworkVector(new double[] { 1, 2, 3 });
            NetworkVector bpVector = new NetworkVector(new double[] { 5, 7, 11 });

            bank.Run(inputVector);
            NetworkVector inputGradientCheck = new NetworkVector(new double[] { 5, 7, 11 });
            Assert.AreEqual(inputGradientCheck, bank.InputGradient(bpVector));


            bank.BackPropagate(bpVector);
            bank.Update(new GradientDescent());

            NetworkMatrix weightsCheck = new NetworkMatrix( new double[,] { { -51 } } );
            NetworkVector biasCheck = new NetworkVector( new double[] { -23 } );

            Assert.AreEqual(biasCheck, bank.Biases);
            Assert.AreEqual(weightsCheck, bank.Weights);
            
        }

        [TestMethod]
        public void CannotBackPropagateWeightedCombinerBankWithBadOutputGradientSize()
        {
            int numberOfBanks = 3;
            WeightedCombiner layer = new WeightedCombiner(new NetworkMatrix( new double[,] { { 1 } }) );
            WeightedCombinerBank bank = new WeightedCombinerBank(layer, numberOfBanks);
            NetworkVector badGradient = new NetworkVector(2);

            try
            {
                bank.BackPropagate(badGradient);
                Assert.Fail("WeightedCombinerBank.BackPropagate failed to throw an ArgumentException for outputgradient of the wrong size.");
            }
            catch (ArgumentException) { }
        }

        //[TestMethod]
        //public void CorrectBackPropagattionOneInput()
        //{
        //    int numberOfBanks = 3;
        //    WeightedCombiner layer = new WeightedCombiner(new double[,] { { 1 } });
        //    WeightedCombinerBank bank = new WeightedCombinerBank(layer, numberOfBanks);
        //    NetworkVector input = new NetworkVector(new double[] { 2, 3, 5 });
        //    NetworkVector outputGradients = new NetworkVector(new double[] { 2, 3, 5 });
        //    NetworkVector inputGradientCheck = new NetworkVector(new double[] { 2, 3, 5 });

        //    bank.Run(input);
        //    bank.BackPropagate(outputGradients);
        //    Assert.AreEqual(inputGradientCheck, bank.InputGradient);
        //}

        //[TestMethod]
        //public void CorrectRunThreeInputs()
        //{
        //    int numberOfBanks = 3;
        //    WeightedCombiner layer = new WeightedCombiner(new double[,] { { 2, 3, 5 } });
        //    WeightedCombinerBank bank = new WeightedCombinerBank(layer, numberOfBanks);
        //    NetworkVector input = new NetworkVector(new double[] { 7, 7, 7, 11, 11, 11, 13, 13, 13 });
        //    NetworkVector outputcheck = new NetworkVector(new double[] { 70, 110, 130 });

        //    bank.Run(input);
        //    Assert.AreEqual(outputcheck, bank.Output);
        //}
    }
}
