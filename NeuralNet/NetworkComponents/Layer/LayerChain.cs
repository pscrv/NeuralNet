using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNet
{
    public class LayerChain : LayerComponent
    {
        #region protected members
        protected _layerListNode _head;
        protected _layerListNode _tail;
        #endregion

        #region constructors
        public LayerChain()
        {
            _head = null;
            _tail = null;
        }

        public LayerChain(Layer layer)
            : this()
        {
            Add(layer);
        }
        #endregion



        #region public properties
        public override int NumberOfInputs { get { return _head.Layer.NumberOfInputs; } }
        public override NetworkVector Output { get { return _tail.Layer.Output; } }
        public override NetworkVector InputGradient { get { return _head.Layer.InputGradient; } }
        public override int NumberOfOutputs { get { return _tail.Layer.NumberOfOutputs; } }

        public int NumberOfLayers
        {
            get
            {
                if (_head == null)
                    return 0;

                if (_head == _tail)
                    return 1;

                int count = 0;
                _layerListNode node = _head;
                while (node != null)
                {
                    count++;
                    node = node.Next;
                }
                return count;

            }
        }
       
        public List<LayerState> State
        {
            get
            {
                List<LayerState> state = new List<LayerState>();
                foreach (Layer layer in ForwardEnumeration)
                    state.Add(layer.State);
                return state;
            }
        }
        #endregion
        


        #region public methods
        public void Add(Layer layerToAdd)
        {
            if (_layerCannotBeAdded(layerToAdd))
                throw new ArgumentException("Attempt to add with inputs size differnt from output size of the last existing Layer");

            _layerListNode nodeToAdd = new _layerListNode(layerToAdd);

            if (_head == null)
            {
                _head = nodeToAdd;
                _tail = _head;
            }
            else
            {
                _tail.Next = nodeToAdd;
                _tail.Next.Previous = _tail;
                _tail = _tail.Next;
            }
        }

        public override void Run(NetworkVector input)
        {
            if (NumberOfLayers == 0)
                throw new InvalidOperationException("Attempt to run a network with no layers.");

            if (input.Dimension != NumberOfInputs)
                throw new ArgumentException(string.Format("The network accepts {0} inputs, but input has dimension {1}", NumberOfInputs, input.Dimension));

            foreach (Layer layer in ForwardEnumeration)
            {
                layer.Run(input);
                input = layer.Output;
            }
        }

        public override void BackPropagate(NetworkVector outputgradient)
        {
            if (NumberOfLayers == 0)
                throw new InvalidOperationException("Attempt to back propogate in a network with no layers.");

            if (outputgradient.Dimension != _tail.Layer.NumberOfOutputs)
                throw new ArgumentException(string.Format("The network has {0} outputs, but outputgradient has dimension {1}", _tail.Layer.NumberOfOutputs, outputgradient.Dimension));

            foreach (Layer layer in BackwardsEnumeration)
            {
                layer.BackPropagate(outputgradient);
                outputgradient = layer.InputGradient;
            }
        }


        #endregion


        #region private methods
        private bool _layerCannotBeAdded(Layer layerToAdd)
        {
            if (NumberOfLayers == 0)
                return false;

            return _tail.Layer.NumberOfOutputs != layerToAdd.NumberOfInputs;
        }
        #endregion


        #region IEnumerable
        public IEnumerable<Layer> ForwardEnumeration
        {
            get
            {
                _layerListNode node = _head;

                while (node != null)
                {
                    yield return node.Layer;
                    node = node.Next;
                }
            }
        }

        public IEnumerable<Layer> BackwardsEnumeration
        {
            get
            {
                _layerListNode node = _tail;

                while (node != null)
                {
                    yield return node.Layer;
                    node = node.Previous;
                }
            }
        }

        #endregion




        #region private subclass of LayerListNode
        protected class _layerListNode
        {
            public Layer Layer { get; protected set; }
            public _layerListNode Next { get; set; }
            public _layerListNode Previous { get;  set; }

            public _layerListNode(Layer layer)
            {
                Layer = layer;
            }
        }
        #endregion
    }


}
