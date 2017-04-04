using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNet
{
    public abstract class BatchPairCollection : IEnumerable<BatchPair>
    {
        #region private attributes
        private BatchPairNode _head;
        private BatchPairNode _tail;

        private IEnumerable<BatchPair> _enumeration;
        #endregion

        #region constructors
        public BatchPairCollection()
        {
            Clear();
        }

        public BatchPairCollection(IEnumerable<BatchPair> enumeration)
        {
            _enumeration = enumeration;
        }
        #endregion

        #region public properties
        public int Count
        {
            get
            {
                int count = 0;
                foreach (BatchPair vp in this)
                {
                    count++;
                }
                return count;
            }
        }
        #endregion

        #region public methods
        public void Add(BatchPair pairToAdd)
        {
            if (pairToAdd == null)
                throw new ArgumentException("Attempt to add a null pair.");

            BatchPairNode newNode = new BatchPairNode(pairToAdd);

            if (_head == null)
            {
                _head = newNode;
                _tail = _head;
            }
            else
            {
                _tail.Next = newNode;
                _tail = newNode;
            }
        }

        public void Clear()
        {
            _head = null;
            _tail = null;
            _enumeration = null;
        }
        #endregion


        #region IEnumerable
        public IEnumerator<BatchPair> GetEnumerator()
        {
            if (_enumeration != null)
            {
                foreach (BatchPair pair in _enumeration)
                {
                    yield return pair;
                }
            }

            BatchPairNode node = _head;
            while (node != null)
            {
                yield return node.Pair;
                node = node.Next;
            }
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return this.GetEnumerator();
        }
        #endregion


        private class BatchPairNode
        {
            public BatchPair Pair { get; }
            public BatchPairNode Next { get; set; }

            public BatchPairNode(BatchPair pair)
            {
                Pair = pair;
                Next = null;
            }
        }
    }

    public class TrainingBatchCollection : BatchPairCollection
    {
        public TrainingBatchCollection()
            : base() { }

        public TrainingBatchCollection(IEnumerable<BatchPair> enumerator)
            : base(enumerator) { }

        public IEnumerable<TrainingBatchCollection> AsSingletons()
        {
            foreach (BatchPair pair in this)
            {
                yield return new TrainingBatchCollection { pair };
            }
        }

        public IEnumerable<TrainingBatchCollection> AsBatches(int batchsize)
        {
            if (batchsize < 1)
                throw new ArgumentException("Attempt to creat batches with < 1 elements each.");

            int count = 0;
            TrainingBatchCollection workingCollection = new TrainingBatchCollection();
            foreach (BatchPair pair in this)
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
