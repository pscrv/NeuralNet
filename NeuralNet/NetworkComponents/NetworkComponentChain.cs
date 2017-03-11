using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNet
{

    public class NetworkComponentChain : NetworkComponent
    {
        #region protected members
        protected _networkComponentNode _head;
        protected _networkComponentNode _tail;
        #endregion

        #region constructors
        public NetworkComponentChain()
        {
            _head = null;
            _tail = null;
        }

        public NetworkComponentChain(NetworkComponent component)
            : this()
        {
            Add(component);
        }
        #endregion



        #region public properties
        public override int NumberOfInputs { get { return _head.Component.NumberOfInputs; } }
        public override NetworkVector Output { get { return _tail.Component.Output; } }
        public override NetworkVector InputGradient { get { return _head.Component.InputGradient; } }
        public override int NumberOfOutputs { get { return _tail.Component.NumberOfOutputs; } }

        public int NumberOfComponents
        {
            get
            {
                if (_head == null)
                    return 0;

                if (_head == _tail)
                    return 1;

                int count = 0;
                _networkComponentNode node = _head;
                while (node != null)
                {
                    count++;
                    node = node.Next;
                }
                return count;

            }
        }
        #endregion



        #region public methods
        public virtual void Add(NetworkComponent componentToAdd)
        {
            if (_componentHasWrongSize(componentToAdd))
                throw new ArgumentException("Attempt to add with inputs size differnt from output size of the last existing Layer");

            _networkComponentNode nodeToAdd = new _networkComponentNode(componentToAdd);

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
            if (NumberOfComponents == 0)
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
            if (NumberOfComponents == 0)
                throw new InvalidOperationException("Attempt to back propogate in a network with no layers.");

            if (outputgradient.Dimension != _tail.Component.NumberOfOutputs)
                throw new ArgumentException(string.Format("The network has {0} outputs, but outputgradient has dimension {1}", _tail.Component.NumberOfOutputs, outputgradient.Dimension));

            foreach (NetworkComponent component in BackwardsEnumeration)
            {
                component.BackPropagate(outputgradient);
                outputgradient = component.InputGradient;
            }
        }


        #endregion


        #region protected and private methods
        protected bool _componentHasWrongSize(NetworkComponent componentToAdd)
        {
            if (NumberOfComponents == 0)
                return false;

            return _tail.Component.NumberOfOutputs != componentToAdd.NumberOfInputs;
        }
        #endregion


        #region IEnumerable
        public IEnumerable<NetworkComponent> ForwardEnumeration
        {
            get
            {
                _networkComponentNode node = _head;

                while (node != null)
                {
                    yield return node.Component;
                    node = node.Next;
                }
            }
        }

        public IEnumerable<NetworkComponent> BackwardsEnumeration
        {
            get
            {
                _networkComponentNode node = _tail;

                while (node != null)
                {
                    yield return node.Component;
                    node = node.Previous;
                }
            }
        }

        #endregion




        #region private subclass of LayerListNode
        protected class _networkComponentNode
        {
            public NetworkComponent Component { get; protected set; }
            public _networkComponentNode Next { get; set; }
            public _networkComponentNode Previous { get; set; }

            public _networkComponentNode(NetworkComponent component)
            {
                Component = component;
            }
        }
        #endregion
    }

}
