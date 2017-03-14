using System;
using System.Collections.Generic;
using Microsoft.VisualStudio.TestTools.UnitTesting;

using NeuralNet;
using NeuralNet.NetComponent;

namespace TestNeuralNet
{
    [TestClass]
    public class WCTrainerTests
    {
        [TestMethod]
        public void CanMakeOnline()
        {
            NetworkMatrix matrix = new NetworkMatrix(new double[,] { { 1, 1 } });
            NeuralNet.NetComponent.WeightedCombiner wc = new NeuralNet.NetComponent.WeightedCombiner(matrix);
            List<TrainingVector> trainingVectors = new List<TrainingVector>
            {
                new TrainingVector(
                    new NetworkVector( new double[] { 1, 1 } ),
                    new NetworkVector( new double[] { 1 })
                    )
            };

            WCOnlineTrainer trainer = new WCOnlineTrainer(wc, trainingVectors);
            Assert.AreNotEqual(null, trainer);
        }

        [TestMethod]
        public void CanRunOnline()
        {
            NetworkMatrix matrix = new NetworkMatrix(new double[,] { { 1, 1 } });
            NeuralNet.NetComponent.WeightedCombiner wc = new NeuralNet.NetComponent.WeightedCombiner(matrix);
            List<TrainingVector> trainingVectors = new List<TrainingVector>
            {
                new TrainingVector(
                    new NetworkVector( new double[] { 0, 0 } ),
                    new NetworkVector( new double[] { 1 })
                    )
            };

            WCOnlineTrainer trainer = new WCOnlineTrainer(wc, trainingVectors);
            trainer.Train();
        }

        [TestMethod]
        public void TrainOnline_CorrectOnePass()
        {
            NetworkMatrix matrix = new NetworkMatrix(new double[,] { { 1, 1 } });
            NeuralNet.NetComponent.WeightedCombiner wc = new NeuralNet.NetComponent.WeightedCombiner(matrix);
            List<TrainingVector> trainingVectors = new List<TrainingVector>
            {
                new TrainingVector(
                    new NetworkVector( new double[] { 0, 0 } ), 
                    new NetworkVector( new double[] { 1 })
                    ),
                new TrainingVector(
                    new NetworkVector( new double[] { 1, 0 } ),
                    new NetworkVector( new double[] { 0 })
                    ),
                new TrainingVector(
                    new NetworkVector( new double[] { 0, 1 } ),
                    new NetworkVector( new double[] { 0 })
                    ),
                new TrainingVector(
                    new NetworkVector( new double[] { 1, 1 } ),
                    new NetworkVector( new double[] { 1 })
                    )
            };

            WCOnlineTrainer trainer = new WCOnlineTrainer(wc, trainingVectors);
            trainer.Train();
            
            NetworkMatrix weightsCheck = new NetworkMatrix(new double[,] { { 1, 3 } });
            NetworkVector biasesCheck = new NetworkVector(new double[] { 1 });

            Assert.AreEqual(biasesCheck, wc.Biases);
            Assert.AreEqual(weightsCheck, wc.Weights);
        }

        [TestMethod]
        public void TrainOnline_CorrectTwoPasses()
        {
            NetworkMatrix matrix = new NetworkMatrix(new double[,] { { 1, 1 } });
            NeuralNet.NetComponent.WeightedCombiner wc = new NeuralNet.NetComponent.WeightedCombiner(matrix);
            List<TrainingVector> trainingVectors = new List<TrainingVector>
            {
                new TrainingVector(
                    new NetworkVector( new double[] { 0, 0 } ),
                    new NetworkVector( new double[] { 1 })
                    ),
                new TrainingVector(
                    new NetworkVector( new double[] { 1, 0 } ),
                    new NetworkVector( new double[] { 0 })
                    ),
                new TrainingVector(
                    new NetworkVector( new double[] { 0, 1 } ),
                    new NetworkVector( new double[] { 0 })
                    ),
                new TrainingVector(
                    new NetworkVector( new double[] { 1, 1 } ),
                    new NetworkVector( new double[] { 1 })
                    )
            };

            WCOnlineTrainer trainer = new WCOnlineTrainer(wc, trainingVectors);
            trainer.Train();
            trainer.Train();

            NetworkMatrix weightsCheck = new NetworkMatrix( new double[,] { { 3, 5 } } );
            NetworkVector biasesCheck = new NetworkVector( new double[] { 1 } );

            Assert.AreEqual(biasesCheck, wc.Biases);
            Assert.AreEqual(weightsCheck, wc.Weights);
        }
        
        [TestMethod]
        public void TrainOnline_CorrectThreePasses()
        {
            NetworkMatrix matrix = new NetworkMatrix(new double[,] { { 1, 1 } });
            NeuralNet.NetComponent.WeightedCombiner wc = new NeuralNet.NetComponent.WeightedCombiner(matrix);
            List<TrainingVector> trainingVectors = new List<TrainingVector>
            {
                new TrainingVector(
                    new NetworkVector( new double[] { 0, 0 } ),
                    new NetworkVector( new double[] { 1 })
                    ),
                new TrainingVector(
                    new NetworkVector( new double[] { 1, 0 } ),
                    new NetworkVector( new double[] { 0 })
                    ),
                new TrainingVector(
                    new NetworkVector( new double[] { 0, 1 } ),
                    new NetworkVector( new double[] { 0 })
                    ),
                new TrainingVector(
                    new NetworkVector( new double[] { 1, 1 } ),
                    new NetworkVector( new double[] { 1 })
                    )
            };

            WCOnlineTrainer trainer = new WCOnlineTrainer(wc, trainingVectors);
            trainer.Train();
            trainer.Train();
            trainer.Train();

            NetworkMatrix weightsCheck = new NetworkMatrix( new double[,] { { 3, 7 } } );
            NetworkVector biasesCheck = new NetworkVector( new double[] { -1 } );

            Assert.AreEqual(biasesCheck, wc.Biases);
            Assert.AreEqual(weightsCheck, wc.Weights);            
        }
        

        [TestMethod]
        public void CanMakeBatchTrainer()
        {
            NetworkMatrix matrix = new NetworkMatrix(new double[,] { { 1, 1 } });
            NeuralNet.NetComponent.WeightedCombiner wc = new NeuralNet.NetComponent.WeightedCombiner(matrix);
            List<TrainingVector> trainingVectors = new List<TrainingVector>
            {
                new TrainingVector(
                    new NetworkVector( new double[] { 1, 1 } ),
                    new NetworkVector( new double[] { 1 })
                    )
            };

            WCBatchTrainer trainer = new WCBatchTrainer(wc, trainingVectors);
            Assert.AreNotEqual(null, trainer);
        }

        [TestMethod]
        public void CanRunBatchTrainer()
        {
            NetworkMatrix matrix = new NetworkMatrix(new double[,] { { 1, 1 } });
            NetworkVector vector = new NetworkVector(new double[] { 0 });
            NeuralNet.NetComponent.WeightedCombiner wc = new NeuralNet.NetComponent.WeightedCombiner(matrix, vector);
            List<TrainingVector> trainingVectors = new List<TrainingVector>
            {
                new TrainingVector(
                    new NetworkVector( new double[] { 0, 0 } ),
                    new NetworkVector( new double[] { 1 })
                    )
            };

            WCBatchTrainer trainer = new WCBatchTrainer(wc, trainingVectors);
            trainer.Train();
        }

        [TestMethod]
        public void BatchTrainCorrectOnePass()
        {
            NetworkMatrix matrix = new NetworkMatrix(new double[,] { { 1, 1 } });
            NeuralNet.NetComponent.WeightedCombiner wc = new NeuralNet.NetComponent.WeightedCombiner(matrix);            
            List<TrainingVector> trainingVectors = new List<TrainingVector>
            {
                new TrainingVector(
                    new NetworkVector( new double[] { 0, 0 } ),
                    new NetworkVector( new double[] { 1 })
                    ),
                new TrainingVector(
                    new NetworkVector( new double[] { 1, 0 } ),
                    new NetworkVector( new double[] { 0 })
                    ),
                new TrainingVector(
                    new NetworkVector( new double[] { 0, 1 } ),
                    new NetworkVector( new double[] { 0 })
                    ),
                new TrainingVector(
                    new NetworkVector( new double[] { 1, 1 } ),
                    new NetworkVector( new double[] { 1 })
                    )
            };

            WCBatchTrainer trainer = new WCBatchTrainer(wc, trainingVectors);
            trainer.Train();

            NetworkMatrix weightsCheck = new NetworkMatrix( new double[,] { { -1, -1 } } );
            NetworkVector biasesCheck = new NetworkVector( new double[] { -2 });
            
            Assert.AreEqual(biasesCheck, wc.Biases);
            Assert.AreEqual(weightsCheck, wc.Weights);
        }

        [TestMethod]
        public void BatchTrainCorrectTwoPasses()
        {
            NetworkMatrix matrix = new NetworkMatrix(new double[,] { { 1, 1 } });
            NeuralNet.NetComponent.WeightedCombiner wc = new NeuralNet.NetComponent.WeightedCombiner(matrix);
            List<TrainingVector> trainingVectors = new List<TrainingVector>
            {
                new TrainingVector(
                    new NetworkVector( new double[] { 0, 0 } ),
                    new NetworkVector( new double[] { 1 })
                    ),
                new TrainingVector(
                    new NetworkVector( new double[] { 1, 0 } ),
                    new NetworkVector( new double[] { 0 })
                    ),
                new TrainingVector(
                    new NetworkVector( new double[] { 0, 1 } ),
                    new NetworkVector( new double[] { 0 })
                    ),
                new TrainingVector(
                    new NetworkVector( new double[] { 1, 1 } ),
                    new NetworkVector( new double[] { 1 })
                    )
            };

            WCBatchTrainer trainer = new WCBatchTrainer(wc, trainingVectors);
            trainer.Train();
            trainer.Train();

            NetworkMatrix weightsCheck = new NetworkMatrix( new double[,] { { 7, 7 } } );
            NetworkVector biasesCheck = new NetworkVector( new double[] { 12 } );

            Assert.AreEqual(biasesCheck, wc.Biases);
            Assert.AreEqual(weightsCheck, wc.Weights);
        }

        [TestMethod]
        public void BatchTrainCorrectThreePasses()
        {
            NetworkMatrix matrix = new NetworkMatrix(new double[,] { { 1, 1 } });
            NeuralNet.NetComponent.WeightedCombiner wc = new NeuralNet.NetComponent.WeightedCombiner(matrix);
            List<TrainingVector> trainingVectors = new List<TrainingVector>
            {
                new TrainingVector(
                    new NetworkVector( new double[] { 0, 0 } ),
                    new NetworkVector( new double[] { 1 })
                    ),
                new TrainingVector(
                    new NetworkVector( new double[] { 1, 0 } ),
                    new NetworkVector( new double[] { 0 })
                    ),
                new TrainingVector(
                    new NetworkVector( new double[] { 0, 1 } ),
                    new NetworkVector( new double[] { 0 })
                    ),
                new TrainingVector(
                    new NetworkVector( new double[] { 1, 1 } ),
                    new NetworkVector( new double[] { 1 })
                    )
            };

            WCBatchTrainer trainer = new WCBatchTrainer(wc, trainingVectors);
            trainer.Train();
            trainer.Train();
            trainer.Train();

            NetworkMatrix weightsCheck = new NetworkMatrix(new double[,] { { -37, -37 } });
            NetworkVector biasesCheck = new NetworkVector(new double[] { -62 });

            Assert.AreEqual(biasesCheck, wc.Biases);
            Assert.AreEqual(weightsCheck, wc.Weights);
        }
    }
}

