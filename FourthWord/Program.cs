using System;
using System.Diagnostics;

using NeuralNet;
using System.Linq;

namespace FourthWord
{
    class Program
    {
        static void Main(string[] args)
        {
            FourthWordNetwork network = new FourthWordNetwork();
            DataReader reader = new DataReader();
            Trainer trainer = new Trainer(network, new SoftMaxWithCrossEntropy(), new GradientDescent(0.00001, 100));

            Stopwatch sw = new Stopwatch();
            Console.WriteLine("Starting training, one epoch, in batches of size 100.");

            TimeSpan lastTotalElapsed = new TimeSpan();
            TimeSpan totalElapsed = new TimeSpan();
            sw.Start();
            int count = 0;
            foreach (TrainingCollection batch in reader.TrainingDataCollection.AsBatches(100))
            {
                VectorPair[] x = batch.ToArray();

                //trainer.ParallelTrain(batch);
                trainer.Train(batch);
                totalElapsed = sw.Elapsed;

                Console.Write(
                    string.Format(
                        "   batch {0}   Cost {1}   <{2}>\r", 
                        count, 
                        trainer.Cost,
                        totalElapsed - lastTotalElapsed)
                    );
                lastTotalElapsed = totalElapsed;
                count++;
            }
           

            Console.ReadLine();
        }
    }
}
