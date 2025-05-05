namespace TurtleML;

using System;
using System.Collections;
using System.Collections.Generic;
using System.Numerics;
using System.Runtime.CompilerServices;

/// <summary>
/// Represents a multi-dimensional tensor of floating-point values.
/// </summary>
public class Tensor : IEnumerable<float>
{
    /// <summary>
    /// An empty tensor with zero elements.
    /// </summary>
    public static readonly Tensor Empty = new (0);

    private readonly float[] values;

    /// <summary>
    /// Initializes a new instance of the <see cref="Tensor"/> class with specified dimensions.
    /// </summary>
    /// <param name="width">The width of the tensor.</param>
    /// <param name="height">The height of the tensor.</param>
    /// <param name="depth">The depth of the tensor.</param>
    public Tensor(int width, int height, int depth)
    {
        Width = width;
        Height = height;
        Depth = depth;

        values = new float[width * height * depth];
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="Tensor"/> class with specified dimensions.
    /// </summary>
    /// <param name="size">A tuple containing width, height, and depth.</param>
    public Tensor((int width, int height, int depth) size)
        : this(size.width, size.height, size.depth)
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="Tensor"/> class with a single dimension.
    /// </summary>
    /// <param name="length">The length of the tensor (width).</param>
    public Tensor(int length)
        : this(length, 1, 1)
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="Tensor"/> class with 2D dimensions.
    /// </summary>
    /// <param name="width">The width of the tensor.</param>
    /// <param name="height">The height of the tensor.</param>
    public Tensor(int width, int height)
        : this(width, height, 1)
    {
    }

    private Tensor(float[] values)
    {
        this.values = values;

        Width = values.Length;
        Height = 1;
        Depth = 1;
    }

    /// <summary>
    /// Gets the tensor depth.
    /// </summary>
    public int Depth { get; private set; }

    /// <summary>
    /// Gets the tensor dimensions in width,height,depth format.
    /// </summary>
    public (int, int, int) Dimensions => (Width, Height, Depth);

    /// <summary>
    /// Gets the tensor height.
    /// </summary>
    public int Height { get; private set; }

    /// <summary>
    /// Gets the tensor total length.
    /// </summary>
    public int Length => values.Length;

    /// <summary>
    /// Gets the tensor width.
    /// </summary>
    public int Width { get; private set; }

    /// <summary>
    /// Gets a reference to the element at the specified index.
    /// </summary>
    /// <param name="i">The index of the element.</param>
    public ref float this[int i] => ref values[i];

    /// <summary>
    /// Gets a reference to the element at the specified 2D coordinates.
    /// </summary>
    /// <param name="x">X coordinate.</param>
    /// <param name="y">Y coordinate.</param>
    public ref float this[int x, int y] => ref this[IndexOf(x, y)];

    /// <summary>
    /// Gets a reference to the element at the specified 3D coordinates.
    /// </summary>
    /// <param name="x">X coordinate.</param>
    /// <param name="y">Y coordinate.</param>
    /// <param name="z">Z coordinate.</param>
    public ref float this[int x, int y, int z] => ref this[IndexOf(x, y, z)];

    /// <summary>
    /// Add the supplied tensors together and return the result.
    /// </summary>
    /// <param name="tensor1">The first tensor to add.</param>
    /// <param name="tensor2">The second tensor to add.</param>
    /// <returns>A new tensor sum of the supplied values.</returns>
    public static Tensor Add(Tensor tensor1, Tensor tensor2) => Add(tensor1, tensor2.values);

    /// <summary>
    /// Adds an array to a tensor element-wise and returns the result.
    /// </summary>
    /// <param name="tensor">The tensor to add to.</param>
    /// <param name="array">The array to add.</param>
    /// <returns>A new tensor containing the sum of the elements.</returns>
    public static Tensor Add(Tensor tensor, float[] array)
    {
        var result = new Tensor(tensor.Dimensions);

        int i = 0,
            step = Vector<float>.Count,
            count = tensor.Length;

        for (; i < count - step; i += step)
        {
            var vector1 = new Vector<float>(tensor.values, i);
            var vector2 = new Vector<float>(array, i);

            Vector.Add(vector1, vector2).CopyTo(result.values, i);
        }

        for (; i < count; i++)
        {
            result[i] = tensor.values[i] + array[i];
        }

        return result;
    }

    /// <summary>
    /// Copies data from a source tensor to a destination tensor.
    /// </summary>
    /// <param name="src">The source tensor.</param>
    /// <param name="srcX">X coordinate in the source tensor.</param>
    /// <param name="srcY">Y coordinate in the source tensor.</param>
    /// <param name="srcZ">Z coordinate in the source tensor.</param>
    /// <param name="dst">The destination tensor.</param>
    /// <param name="dstX">X coordinate in the destination tensor.</param>
    /// <param name="dstY">Y coordinate in the destination tensor.</param>
    /// <param name="dstZ">Z coordinate in the destination tensor.</param>
    /// <param name="count">Number of elements to copy.</param>
    public static void Copy(Tensor src, int srcX, int srcY, int srcZ, Tensor dst, int dstX, int dstY, int dstZ, int count)
    {
        var srcOffset = src.IndexOf(srcX, srcY, srcZ);
        var dstOffset = dst.IndexOf(dstX, dstY, dstZ);

        Array.Copy(src.values, srcOffset, dst.values, dstOffset, count);
    }

    /// <summary>
    /// Creates a tensor from an existing float array.
    /// </summary>
    /// <param name="array">The input float array.</param>
    /// <returns>A new tensor containing the array data.</returns>
    public static Tensor Create(float[] array)
    {
        var tensor = new Tensor(array.Length);
        tensor.Load(array);
        return tensor;
    }

    /// <summary>
    /// Creates a copy of an existing tensor.
    /// </summary>
    /// <param name="tensor">The tensor to copy.</param>
    /// <returns>A new tensor with the same data as the input.</returns>
    public static Tensor Create(Tensor tensor)
    {
        return Create(tensor.values);
    }

    /// <summary>
    /// Computes the dot product of two tensors.
    /// </summary>
    /// <param name="tensor1">The first tensor.</param>
    /// <param name="tensor2">The second tensor.</param>
    /// <returns>The dot product of the two tensors.</returns>
    public static float Dot(Tensor tensor1, Tensor tensor2) => Dot(tensor1.values, tensor2.values);

    /// <summary>
    /// Computes the dot product of two arrays.
    /// </summary>
    /// <param name="array1">The first array.</param>
    /// <param name="array2">The second array.</param>
    /// <returns>The dot product of the two arrays.</returns>
    public static float Dot(float[] array1, float[] array2)
    {
        float accumulator = 0f;

        int i = 0,
            step = Vector<float>.Count,
            count = Math.Min(array1.Length, array2.Length);

        for (; i < count - step; i += step)
        {
            var vector1 = new Vector<float>(array1, i);
            var vector2 = new Vector<float>(array2, i);

            accumulator += Vector.Dot(vector1, vector2);
        }

        for (; i < count; i++)
        {
            accumulator += array1[i] * array2[i];
        }

        return accumulator;
    }

    /// <summary>
    /// Computes the dot product of two spans.
    /// </summary>
    /// <param name="span1">The first span.</param>
    /// <param name="span2">The second span.</param>
    /// <returns>The sum of element-wise products of the spans.</returns>
    public static float Dot(Span<float> span1, Span<float> span2)
    {
        var vector1 = new Vector<float>(span1);
        var vector2 = new Vector<float>(span2);

        return Vector.Dot(vector1, vector2);
    }

    /// <summary>
    /// Multiplies the tensor by a scalar value and returns a new tensor.
    /// </summary>
    /// <param name="tensor">The input tensor.</param>
    /// <param name="value">The scalar multiplier.</param>
    /// <returns>A new tensor with all elements multiplied by the scalar.</returns>
    public static Tensor Multiply(Tensor tensor, float value)
    {
        var result = new Tensor(tensor.Dimensions);
        Multiply(tensor, value, result);
        return result;
    }

    /// <summary>
    /// Multiplies the tensor by a scalar value in-place.
    /// </summary>
    /// <param name="tensor">The input tensor.</param>
    /// <param name="value">The scalar multiplier.</param>
    /// <param name="result">The output tensor to store results.</param>
    public static void Multiply(Tensor tensor, float value, Tensor result)
    {
        int i = 0,
            step = Vector<float>.Count,
            count = tensor.Length;

        for (; i < count - step; i += step)
        {
            var vector = new Vector<float>(tensor.values, i);

            Vector.Multiply(vector, value).CopyTo(result.values, i);
        }

        for (; i < count; i++)
        {
            result[i] = tensor.values[i] * value;
        }
    }

    /// <summary>
    /// Multiplies two tensors element-wise in-place.
    /// </summary>
    /// <param name="tensor">The input tensor.</param>
    /// <param name="array">The array to multiply with.</param>
    /// <param name="result">The output tensor to store results.</param>
    public static void Multiply(Tensor tensor, float[] array, Tensor result)
    {
        int i = 0,
            step = Vector<float>.Count,
            count = tensor.Length;

        for (; i < count - step; i += step)
        {
            var vector1 = new Vector<float>(tensor.values, i);
            var vector2 = new Vector<float>(array, i);

            Vector.Multiply(vector1, vector2).CopyTo(result.values, i);
        }

        for (; i < count; i++)
        {
            result[i] = tensor.values[i] * array[i];
        }
    }

    /// <summary>
    /// Multiplies two tensors element-wise and returns a new tensor.
    /// </summary>
    /// <param name="tensor1">The first input tensor.</param>
    /// <param name="tensor2">The second input tensor.</param>
    /// <returns>A new tensor with element-wise products of the inputs.</returns>
    public static Tensor Multiply(Tensor tensor1, Tensor tensor2) => Multiply(tensor1, tensor2.values);

    /// <summary>
    /// Multiplies the tensor by an array element-wise and returns a new tensor.
    /// </summary>
    /// <param name="tensor">The input tensor.</param>
    /// <param name="array">The array to multiply with.</param>
    /// <returns>A new tensor with element-wise products of the inputs.</returns>
    public static Tensor Multiply(Tensor tensor, float[] array)
    {
        var result = new Tensor(tensor.Dimensions);
        Multiply(tensor, array, result);
        return result;
    }

    /// <summary>
    /// Wraps an existing float array into a tensor without copying.
    /// </summary>
    /// <param name="values">The input float array.</param>
    /// <returns>A tensor that references the provided array.</returns>
    public static Tensor Wrap(float[] values)
    {
        return new Tensor(values);
    }

    /// <summary>
    /// Adds a tensor to this tensor and returns a new tensor with the result.
    /// </summary>
    /// <param name="tensor">The tensor to add.</param>
    /// <returns>A new tensor containing the sum of the elements.</returns>
    public Tensor Add(Tensor tensor) => Add(tensor.values);

    /// <summary>
    /// Adds an array to this tensor element-wise and returns a new tensor with the result.
    /// </summary>
    /// <param name="array">The array to add.</param>
    /// <returns>A new tensor containing the sum of the elements.</returns>
    public Tensor Add(float[] array)
    {
        int i = 0,
            step = Vector<float>.Count,
            count = values.Length;

        for (; i < count - step; i += step)
        {
            var vector1 = new Vector<float>(values, i);
            var vector2 = new Vector<float>(array, i);

            Vector.Add(vector1, vector2).CopyTo(values, i);
        }

        for (; i < count; i++)
        {
            values[i] += array[i];
        }

        return this;
    }

    /// <summary>
    /// Clears the tensor values to a specified value.
    /// </summary>
    /// <param name="value">The value to set all elements to.</param>
    public void Clear(float value)
    {
        for (int i = 0, count = values.Length; i < count; i++)
        {
            values[i] = value;
        }
    }

    /// <summary>
    /// Clears the tensor values to zero.
    /// </summary>
    public void Clear()
    {
        Array.Clear(values, 0, values.Length);
    }

    /// <summary>
    /// Copies data from this tensor to another tensor.
    /// </summary>
    /// <param name="tensor">The destination tensor.</param>
    /// <param name="offset">The offset in the destination tensor.</param>
    public void CopyTo(Tensor tensor, int offset) => CopyTo(tensor.values, offset);

    /// <summary>
    /// Copies data from this tensor to a float array.
    /// </summary>
    /// <param name="array">The destination array.</param>
    /// <param name="offset">The offset in the destination array.</param>
    public void CopyTo(float[] array, int offset)
    {
        Array.Copy(values, 0, array, offset, values.Length);
    }

    /// <summary>
    /// Computes the dot product of this tensor with another array.
    /// </summary>
    /// <param name="array">The array to compute the dot product with.</param>
    /// <returns>The dot product result.</returns>
    public float Dot(float[] array) => Dot(values, array);

    /// <summary>
    /// Computes the dot product of this tensor with another tensor.
    /// </summary>
    /// <param name="tensor">The tensor to compute the dot product with.</param>
    /// <returns>The dot product result.</returns>
    public float Dot(Tensor tensor) => Dot(values, tensor.values);

    /// <summary>
    /// Returns an enumerator for the tensor elements.
    /// </summary>
    /// <returns>An enumerator for the tensor elements.</returns>
    public IEnumerator<float> GetEnumerator()
    {
        return ((IEnumerable<float>)values).GetEnumerator();
    }

    /// <summary>
    /// Returns an enumerator for the tensor elements (non-generic version).
    /// </summary>
    /// <returns>An enumerator for the tensor elements.</returns>
    IEnumerator IEnumerable.GetEnumerator()
    {
        return values.GetEnumerator();
    }

    /// <summary>
    /// Gets the index of a 2D coordinate in the underlying array.
    /// </summary>
    /// <param name="x">X coordinate.</param>
    /// <param name="y">Y coordinate.</param>
    /// <returns>The corresponding index in the array.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public int IndexOf(int x, int y)
    {
        return x + (y * Width);
    }

    /// <summary>
    /// Gets the index of a 3D coordinate in the underlying array.
    /// </summary>
    /// <param name="x">X coordinate.</param>
    /// <param name="y">Y coordinate.</param>
    /// <param name="z">Z coordinate.</param>
    /// <returns>The corresponding index in the array.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public int IndexOf(int x, int y, int z)
    {
        return x + (y * Width) + (z * Width * Height);
    }

    /// <summary>
    /// Loads data from a source tensor into this tensor.
    /// </summary>
    /// <param name="source">The source tensor to load data from.</param>
    /// <param name="sourceOffset">The offset in the source tensor's data array.</param>
    public void Load(Tensor source, int sourceOffset) => Load(source.values, sourceOffset);

    /// <summary>
    /// Loads data from a float array into this tensor.
    /// </summary>
    /// <param name="source">The source array to load data from.</param>
    /// <param name="sourceOffset">The offset in the source array.</param>
    public void Load(float[] source, int sourceOffset)
    {
        Buffer.BlockCopy(source, sourceOffset, values, 0, values.Length * sizeof(float));
    }

    /// <summary>
    /// Loads data from a source tensor into this tensor.
    /// </summary>
    /// <param name="source">The source tensor to load data from.</param>
    public void Load(Tensor source) => Load(source.values);

    /// <summary>
    /// Loads data from a float array into this tensor.
    /// </summary>
    /// <param name="source">The source array to load data from.</param>
    public void Load(float[] source)
    {
        Buffer.BlockCopy(source, 0, values, 0, values.Length * sizeof(float));
    }

    /// <summary>
    /// Multiplies the tensor by a scalar value in-place.
    /// </summary>
    /// <param name="value">The scalar multiplier.</param>
    /// <returns>The current tensor reference.</returns>
    public Tensor Multiply(float value)
    {
        Multiply(this, value, this);

        return this;
    }

    /// <summary>
    /// Multiplies the tensor by another tensor element-wise in-place.
    /// </summary>
    /// <param name="tensor">The tensor to multiply with.</param>
    /// <returns>The current tensor reference.</returns>
    public Tensor Multiply(Tensor tensor) => Multiply(tensor.values);

    /// <summary>
    /// Multiplies the tensor by an array element-wise in-place.
    /// </summary>
    /// <param name="array">The array to multiply with.</param>
    /// <returns>The current tensor reference.</returns>
    public Tensor Multiply(float[] array)
    {
        Multiply(this, array, this);

        return this;
    }

    /// <summary>
    /// Reshapes the tensor to the specified dimensions.
    /// </summary>
    /// <param name="width">New width of the tensor.</param>
    /// <param name="height">New height of the tensor.</param>
    /// <param name="depth">New depth of the tensor.</param>
    /// <returns>This tensor with updated dimensions.</returns>
    public Tensor Reshape(int width, int height, int depth)
    {
        Width = width;
        Height = height;
        Depth = depth;

        return this;
    }
}
