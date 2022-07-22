namespace TurtleML;

using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.CompilerServices;

/// <summary>
/// Represents an inference network for calculating an output.
/// </summary>
public class InferenceNetwork
{
    /// <summary>
    /// The collection of layers in the network.
    /// </summary>
    protected readonly ILayer[] Layers;
    /// <summary>
    /// The loss function.
    /// </summary>
    protected readonly ILossFunction Loss;

    /// <summary>
    /// Initializes a new instance of the <see cref="InferenceNetwork"/> class.
    /// </summary>
    /// <param name="loss"></param>
    /// <param name="layers"></param>
    protected InferenceNetwork(ILossFunction loss, ILayer[] layers)
    {
        Loss = loss;
        Layers = layers;
    }

    /// <summary>
    ///
    /// </summary>
    public IEnumerable<ILayer> InternalLayers => Layers;

    /// <summary>
    ///
    /// </summary>
    /// <param name="fileInfo"></param>
    /// <returns></returns>
    public static InferenceNetwork Restore(FileInfo fileInfo)
    {
        using var file = fileInfo.OpenRead();
        return Restore(file);
    }

    /// <summary>
    ///
    /// </summary>
    /// <param name="stream"></param>
    /// <returns></returns>
    /// <exception cref="InvalidDataException"></exception>
    /// <exception cref="InvalidOperationException"></exception>
    public static InferenceNetwork Restore(Stream stream)
    {
        using var reader = new BinaryReader(stream);
        var magicNumber = reader.ReadChars(3);
        if (magicNumber[0] != 't' || magicNumber[1] != 'n' || magicNumber[2] != 'n')
        {
            throw new InvalidDataException("Source data is not a valid neural network dump.");
        }

        var version = reader.ReadInt32();
        if (version != 1)
        {
            throw new InvalidOperationException($"Cannot read version {version} format.");
        }

        var lossFunctionType = reader.ReadString();
        if (!(Activator.CreateInstance(Type.GetType(lossFunctionType)) is ILossFunction lossFunction))
        {
            throw new InvalidOperationException($"An invalid loss function \"{lossFunctionType}\" was specified.");
        }

        var layerCount = reader.ReadInt32();
        var layers = new ILayer[layerCount];
        for (int i = 0; i < layerCount; i++)
        {
            var layerType = reader.ReadString();
            if (RuntimeHelpers.GetUninitializedObject(Type.GetType(layerType)) is not ILayer layer)
            {
                throw new InvalidOperationException($"An invalid layer \"{layerType}\" was specified.");
            }

            layer.Restore(reader);
            layers[i] = layer;
        }

        return new InferenceNetwork(lossFunction, layers);
    }

    /// <summary>
    ///
    /// </summary>
    /// <param name="inputs"></param>
    /// <returns></returns>
    public Tensor CalculateOutputs(Tensor inputs)
    {
        var results = inputs;
        for (int l = 0, count = Layers.Length; l < count; l++)
        {
            results = Layers[l].CalculateOutputs(results, training: false);
        }

        return results;
    }
}