using System;
using System.Collections.Generic;

namespace NeuralNet
{
    public class TrainingCollection : List<VectorPair>
    {
        public TrainingCollection(int capacity)
            : base(capacity) { }

        public TrainingCollection()
            : base() { }


        public IEnumerable<TrainingCollection> AsSingletons()
        {
            foreach (VectorPair pair in this)
            {
                yield return new TrainingCollection { pair };
            }
        }

        public IEnumerable<TrainingCollection> AsBatches(int batchsize)
        {
            if (batchsize < 1)
                throw new ArgumentException("Attempt to creat batches with < 1 elements each.");

            int count = 0;
            TrainingCollection workingCollection = new TrainingCollection(batchsize);
            foreach (VectorPair pair in this)
            {
                if (count == batchsize)
                {
                    yield return workingCollection;
                    workingCollection.Clear();
                    count = 0;
                }

                workingCollection.Add(pair);
                count++;
                
            }

            yield return workingCollection;
        }
    }
}
