using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;

namespace NeuralNet
{
    public abstract class VectorPairCollection : IEnumerable<VectorPair>
    {
        #region private attributes
        private VectorPairNode _head;
        private VectorPairNode _tail;

        private IEnumerable<VectorPair> _enumeration;
        #endregion

        #region constructors
        public VectorPairCollection()
        {
            Clear();
        }

        public VectorPairCollection(IEnumerable<VectorPair> enumeration)
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
                foreach (VectorPair vp in this)
                {
                    count++;
                }
                return count;
            }
        }
        #endregion

        #region public methods
        public void Add(VectorPair pairToAdd)
        {
            if (pairToAdd == null)
                throw new ArgumentException("Attempt to add a null pair.");

            VectorPairNode newNode = new VectorPairNode(pairToAdd);

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
        public IEnumerator<VectorPair> GetEnumerator()
        {
            if (_enumeration != null)
            {
                foreach (VectorPair pair in _enumeration)
                {
                    yield return pair;
                }
            }

            VectorPairNode node = _head;
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


        private class VectorPairNode
        {
            public VectorPair Pair { get; }
            public VectorPairNode Next { get; set; }

            public VectorPairNode(VectorPair pair)
            {
                Pair = pair;
                Next = null;
            }
        }
    }


    public class TrainingCollection : VectorPairCollection
    {
        public TrainingCollection()
            : base() { }

        public TrainingCollection(IEnumerable<VectorPair> enumerator)
            : base(enumerator) { }

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
            TrainingCollection workingCollection = new TrainingCollection();
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
