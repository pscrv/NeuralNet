using System;
using System.Collections.Generic;
using Microsoft.VisualStudio.TestTools.UnitTesting;

using NeuralNet;

namespace TestNeuralNet
{
    [TestClass]
    public class TrainerTests
    {
        [TestMethod]
        public void CanMakeOnline()
        {
            NetworkMatrix matrix = new NetworkMatrix(new double[,] { { 1, 1 } });
            WeightedCombiner wc = new WeightedCombiner(matrix);
            List<TrainingVector> trainingVectors = new List<TrainingVector>
            {
                new TrainingVector(
                    new NetworkVector( new double[] { 1, 1 } ),
                    new NetworkVector( new double[] { 1 })
                    )
            };

            OnlineTrainer trainer = new OnlineTrainer(wc, trainingVectors);
            Assert.AreNotEqual(null, trainer);
        }

        [TestMethod]
        public void CanRunOnline_WC()
        {
            NetworkMatrix matrix = new NetworkMatrix(new double[,] { { 1, 1 } });
            WeightedCombiner wc = new WeightedCombiner(matrix);
            List<TrainingVector> trainingVectors = new List<TrainingVector>
            {
                new TrainingVector(
                    new NetworkVector( new double[] { 0, 0 } ),
                    new NetworkVector( new double[] { 1 })
                    )
            };
        }

        [TestMethod]
        public void CanRunOnline_LinearLayer()
        {
            NetworkMatrix matrix = new NetworkMatrix(new double[,] { { 1, 1 } });
            Layer layer = Layer.CreateLinearLayer(matrix);
            List<TrainingVector> trainingVectors = new List<TrainingVector>
            {
                new TrainingVector(
                    new NetworkVector( new double[] { 0, 0 } ),
                    new NetworkVector( new double[] { 1 })
                    )
            };

            OnlineTrainer trainer = new OnlineTrainer(layer, trainingVectors);
            trainer.Train();
        }

        [TestMethod]
        public void CanRunOnline_LogisticLayer()
        {
            NetworkMatrix matrix = new NetworkMatrix(new double[,] { { 1, 1 } });
            Layer layer = Layer.CreateLogisticLayer(matrix);
            List<TrainingVector> trainingVectors = new List<TrainingVector>
            {
                new TrainingVector(
                    new NetworkVector( new double[] { 0, 0 } ),
                    new NetworkVector( new double[] { 1 })
                    )
            };

            OnlineTrainer trainer = new OnlineTrainer(layer, trainingVectors);
            trainer.Train();
        }

        [TestMethod]
        public void TrainOnline_WC_CorrectOnePass()
        {
            NetworkMatrix matrix = new NetworkMatrix(new double[,] { { 1, 1 } });
            WeightedCombiner wc = new WeightedCombiner(matrix);
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

            OnlineTrainer trainer = new OnlineTrainer(wc, trainingVectors);
            trainer.Train();
            
            NetworkMatrix weightsCheck = new NetworkMatrix(new double[,] { { 1, 3 } });
            NetworkVector biasesCheck = new NetworkVector(new double[] { 1 });

            Assert.AreEqual(biasesCheck, wc.Biases);
            Assert.AreEqual(weightsCheck, wc.Weights);
        }

        [TestMethod]
        public void TrainOnline_LinearLayer_CorrectOnePass()
        {
            NetworkMatrix matrix = new NetworkMatrix(new double[,] { { 1, 1 } });
            Layer layer = Layer.CreateLinearLayer(matrix);
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

            OnlineTrainer trainer = new OnlineTrainer(layer, trainingVectors);
            trainer.Train();

            NetworkMatrix weightsCheck = new NetworkMatrix(new double[,] { { 1, 3 } });
            NetworkVector biasesCheck = new NetworkVector(new double[] { 1 });

            Assert.AreEqual(biasesCheck, layer.Biases);
            Assert.AreEqual(weightsCheck, layer.Weights);
        }
        

        [TestMethod]
        public void TrainOnline_WC_CorrectTwoPasses()
        {
            NetworkMatrix matrix = new NetworkMatrix(new double[,] { { 1, 1 } });
            WeightedCombiner wc = new WeightedCombiner(matrix);
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

            OnlineTrainer trainer = new OnlineTrainer(wc, trainingVectors);
            trainer.Train();
            trainer.Train();

            NetworkMatrix weightsCheck = new NetworkMatrix( new double[,] { { 3, 5 } } );
            NetworkVector biasesCheck = new NetworkVector( new double[] { 1 } );

            Assert.AreEqual(biasesCheck, wc.Biases);
            Assert.AreEqual(weightsCheck, wc.Weights);
        }
        
        [TestMethod]
        public void TrainOnline_WC_CorrectThreePasses()
        {
            NetworkMatrix matrix = new NetworkMatrix(new double[,] { { 1, 1 } });
            WeightedCombiner wc = new WeightedCombiner(matrix);
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

            OnlineTrainer trainer = new OnlineTrainer(wc, trainingVectors);
            trainer.Train();
            trainer.Train();
            trainer.Train();

            NetworkMatrix weightsCheck = new NetworkMatrix( new double[,] { { 3, 7 } } );
            NetworkVector biasesCheck = new NetworkVector( new double[] { -1 } );

            Assert.AreEqual(biasesCheck, wc.Biases);
            Assert.AreEqual(weightsCheck, wc.Weights);            
        }
        

        [TestMethod]
        public void CanMakeBatchTrainer_WC()
        {
            NetworkMatrix matrix = new NetworkMatrix(new double[,] { { 1, 1 } });
            WeightedCombiner wc = new WeightedCombiner(matrix);
            List<TrainingVector> trainingVectors = new List<TrainingVector>
            {
                new TrainingVector(
                    new NetworkVector( new double[] { 1, 1 } ),
                    new NetworkVector( new double[] { 1 })
                    )
            };

            BatchTrainer trainer = new BatchTrainer(wc, trainingVectors);
            Assert.AreNotEqual(null, trainer);
        }

        [TestMethod]
        public void CanMakeBatchTrainer_LinearLayer()
        {
            NetworkMatrix matrix = new NetworkMatrix(new double[,] { { 1, 1 } });
            Layer layer = Layer.CreateLinearLayer(matrix);
            List<TrainingVector> trainingVectors = new List<TrainingVector>
            {
                new TrainingVector(
                    new NetworkVector( new double[] { 1, 1 } ),
                    new NetworkVector( new double[] { 1 })
                    )
            };

            BatchTrainer trainer = new BatchTrainer(layer, trainingVectors);
            Assert.AreNotEqual(null, trainer);
        }

        [TestMethod]
        public void CanMakeBatchTrainer_LogisticsLayer()
        {
            NetworkMatrix matrix = new NetworkMatrix(new double[,] { { 1, 1 } });
            Layer layer = Layer.CreateLogisticLayer(matrix);
            List<TrainingVector> trainingVectors = new List<TrainingVector>
            {
                new TrainingVector(
                    new NetworkVector( new double[] { 1, 1 } ),
                    new NetworkVector( new double[] { 1 })
                    )
            };

            BatchTrainer trainer = new BatchTrainer(layer, trainingVectors);
            Assert.AreNotEqual(null, trainer);
        }

        [TestMethod]
        public void CanRunBatchTrainer_WC()
        {
            NetworkMatrix matrix = new NetworkMatrix(new double[,] { { 1, 1 } });
            NetworkVector vector = new NetworkVector(new double[] { 0 });
            WeightedCombiner wc = new WeightedCombiner(matrix, vector);
            List<TrainingVector> trainingVectors = new List<TrainingVector>
            {
                new TrainingVector(
                    new NetworkVector( new double[] { 0, 0 } ),
                    new NetworkVector( new double[] { 1 })
                    )
            };

            BatchTrainer trainer = new BatchTrainer(wc, trainingVectors);
            trainer.Train();
        }

        [TestMethod]
        public void CanRunBatchTrainer_LinearLayer()
        {
            NetworkMatrix matrix = new NetworkMatrix(new double[,] { { 1, 1 } });
            NetworkVector vector = new NetworkVector(new double[] { 0 });
            Layer layer = Layer.CreateLinearLayer(matrix, vector);
            List<TrainingVector> trainingVectors = new List<TrainingVector>
            {
                new TrainingVector(
                    new NetworkVector( new double[] { 0, 0 } ),
                    new NetworkVector( new double[] { 1 })
                    )
            };

            BatchTrainer trainer = new BatchTrainer(layer, trainingVectors);
            trainer.Train();
        }

        [TestMethod]
        public void CanRunBatchTrainer_LogisticsLayer()
        {
            NetworkMatrix matrix = new NetworkMatrix(new double[,] { { 1, 1 } });
            NetworkVector vector = new NetworkVector(new double[] { 0 });
            Layer layer = Layer.CreateLogisticLayer(matrix, vector);
            List<TrainingVector> trainingVectors = new List<TrainingVector>
            {
                new TrainingVector(
                    new NetworkVector( new double[] { 0, 0 } ),
                    new NetworkVector( new double[] { 1 })
                    )
            };

            BatchTrainer trainer = new BatchTrainer(layer, trainingVectors);
            trainer.Train();
        }

        [TestMethod]
        public void BatchTrainCorrectOnePass_WC()
        {
            NetworkMatrix matrix = new NetworkMatrix(new double[,] { { 1, 1 } });
            WeightedCombiner wc = new WeightedCombiner(matrix);            
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

            BatchTrainer trainer = new BatchTrainer(wc, trainingVectors);
            trainer.Train();

            NetworkMatrix weightsCheck = new NetworkMatrix( new double[,] { { -1, -1 } } );
            NetworkVector biasesCheck = new NetworkVector( new double[] { -2 });
            
            Assert.AreEqual(biasesCheck, wc.Biases);
            Assert.AreEqual(weightsCheck, wc.Weights);
        }

        [TestMethod]
        public void BatchTrainCorrectOnePass_LinearLayer()
        {
            NetworkMatrix matrix = new NetworkMatrix(new double[,] { { 1, 1 } });
            Layer layer = Layer.CreateLinearLayer(matrix);
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

            BatchTrainer trainer = new BatchTrainer(layer, trainingVectors);
            trainer.Train();

            NetworkMatrix weightsCheck = new NetworkMatrix(new double[,] { { -1, -1 } });
            NetworkVector biasesCheck = new NetworkVector(new double[] { -2 });

            Assert.AreEqual(biasesCheck, layer.Biases);
            Assert.AreEqual(weightsCheck, layer.Weights);
        }

        [TestMethod]
        public void BatchTrainCorrectTwoPasses_WC()
        {
            NetworkMatrix matrix = new NetworkMatrix(new double[,] { { 1, 1 } });
            WeightedCombiner wc = new WeightedCombiner(matrix);
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

            BatchTrainer trainer = new BatchTrainer(wc, trainingVectors);
            trainer.Train();
            trainer.Train();

            NetworkMatrix weightsCheck = new NetworkMatrix( new double[,] { { 7, 7 } } );
            NetworkVector biasesCheck = new NetworkVector( new double[] { 12 } );

            Assert.AreEqual(biasesCheck, wc.Biases);
            Assert.AreEqual(weightsCheck, wc.Weights);
        }

        [TestMethod]
        public void BatchTrainCorrectThreePasses_WC()
        {
            NetworkMatrix matrix = new NetworkMatrix(new double[,] { { 1, 1 } });
            WeightedCombiner wc = new WeightedCombiner(matrix);
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

            BatchTrainer trainer = new BatchTrainer(wc, trainingVectors);
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

