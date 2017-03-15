using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNet.NetComponent
{

    public class NetComponentChain : NetComponent
    {
        #region protected members
        protected _networkComponentNode _head;
        protected _networkComponentNode _tail;
        #endregion

        #region constructors
        public NetComponentChain()
        {
            _head = null;
            _tail = null;
        }

        public NetComponentChain(NetComponent component)
            : this()
        {
            AddFixed(component);
        }

        public NetComponentChain(TrainableComponent component)
            : this()
        {
            AddTrainable(component);
        }
        #endregion



        #region NetComponent overrides
        public override int NumberOfInputs { get { return _head.Component.NumberOfInputs; } }
        public override int NumberOfOutputs { get { return _tail.Component.NumberOfOutputs; } }
        public override NetworkVector Output {
            get { return _tail.Component.Output; }
            protected set { }
        }

        public override NetworkVector InputGradient(NetworkVector outputgradient)
        {
            if (NumberOfComponents == 0)
                throw new InvalidOperationException("Attempt to back propogate in a network with no layers.");

            if (outputgradient.Dimension != _tail.Component.NumberOfOutputs)
                throw new ArgumentException(string.Format("The network has {0} outputs, but outputgradient has dimension {1}", _tail.Component.NumberOfOutputs, outputgradient.Dimension));

            NetworkVector gradient = outputgradient.Copy();
            foreach (NetComponent component in BackwardsEnumeration)
            {
                gradient = component.InputGradient(gradient);
            }

            return gradient;
        }        
        #endregion


        #region public methods
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
        public virtual void AddFixed(NetComponent componentToAdd)
        {
            _addComponent(componentToAdd, istrainable: false);
        }

        public virtual void AddTrainable(TrainableComponent componentToAdd)
        {
            _addComponent(componentToAdd, istrainable: true);
        }
        public override void Run(NetworkVector input)
        {
            if (NumberOfComponents == 0)
                throw new InvalidOperationException("Attempt to run a network with no layers.");

            if (input.Dimension != NumberOfInputs)
                throw new ArgumentException(string.Format("The network accepts {0} inputs, but input has dimension {1}", NumberOfInputs, input.Dimension));

            foreach (NetComponent component in ForwardEnumeration)
            {
                component.Run(input);
                input = component.Output;
            }
        }

        #endregion


        #region protected and private methods
        protected virtual void _addComponent(NetComponent componentToAdd, bool istrainable)
        {
            if (_componentHasWrongSize(componentToAdd))
                throw new ArgumentException("Attempt to add with inputs size differnt from output size of the last existing Layer");

            _networkComponentNode nodeToAdd = new _networkComponentNode(componentToAdd, istrainable);

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

        protected bool _componentHasWrongSize(NetComponent componentToAdd)
        {
            if (NumberOfComponents == 0)
                return false;

            return _tail.Component.NumberOfOutputs != componentToAdd.NumberOfInputs;
        }
        #endregion


        #region IEnumerable
        public IEnumerable<NetComponent> ForwardEnumeration
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

        public IEnumerable<NetComponent> ForwardTrainableComponentsEnumeration
        {
            get
            {
                _networkComponentNode node = _head;

                while (node != null)
                {
                    if (node.IsTrainable)
                        yield return node.Component;
                    node = node.Next;
                }
            }
        }

        public IEnumerable<NetComponent> BackwardsEnumeration
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

        public IEnumerable<TrainableComponent> BackwardsTrainableComponentEnumeration
        {
            get
            {
                _networkComponentNode node = _tail;

                while (node != null)
                {
                    if (node.IsTrainable)
                        yield return node.Component as TrainableComponent;
                    node = node.Previous;
                }
            }
        }
        #endregion




        #region private subclass of LayerListNode
        protected class _networkComponentNode
        {
            public NetComponent Component { get; protected set; }
            public bool IsTrainable { get; protected set; }
            public _networkComponentNode Next { get; set; }
            public _networkComponentNode Previous { get; set; }

            public _networkComponentNode(NetComponent component, bool istrainable)
            {
                Component = component;
                IsTrainable = istrainable;
            }
        }
        #endregion
    }

}
