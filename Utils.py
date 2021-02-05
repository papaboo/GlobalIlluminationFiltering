import torch


# Expand the input with one dimension of size dim_size by inserting the new dimension between dim-1 and dim.
# The elements in the tensor is moved to the entries specified in projected_indices. The rest are left as 0.
# Fx the 2D tensor [[4, 1], [2, 3]] projected to 3D using the indices [0,1,0,0] along z would become
# z0: [[4, 0], [2, 3]]
# z1: [[0, 1], [0, 0]]
def project_tensor(input, dim, projected_indices, dim_size):
    assert projected_indices.dtype == torch.long, "projected_indices must have long dtype"

    # Conceptually reshape the tensor to a 2D matrix to make it simpler to project it into the new dimension.
    element_count = 1
    outer_element_count = 1
    for d in range(input.ndim):
        element_count *= input.size()[d]
        if d < dim:
            outer_element_count *= input.size()[d]
    inner_element_count = element_count // outer_element_count

    assert element_count == projected_indices.size()[0], "projected_indices must have the same element count as the input tensor"

    inner_indices = torch.arange(0, inner_element_count, dtype=torch.long)
    inner_indices = inner_indices.repeat(outer_element_count)

    outer_indices = torch.arange(0, outer_element_count, dtype=torch.long)
    outer_indices = outer_indices.repeat_interleave(inner_element_count)

    indices = torch.stack([outer_indices, projected_indices, inner_indices])
    projected_size = torch.Size([outer_element_count, dim_size, inner_element_count])
    sparse_projected_tensor = torch.sparse.FloatTensor(indices, input.view(-1), projected_size)
    dense_projected_tensor = sparse_projected_tensor.to_dense()

    output_size = list(input.size())
    output_size.insert(dim, dim_size)

    return dense_projected_tensor.view(torch.Size(output_size))


def unproject_tensor(input, dim, projected_indices):
    indices_size = list(input.size())
    indices_size[dim] = 1
    projected_indices = projected_indices.view(indices_size)

    gathered_values = torch.gather(input, dim, projected_indices)
    return gathered_values.squeeze(dim)


def test_project_tensor():
    width = 4
    height = 3
    depth = 3
    element_count = depth * height * width

    def verify_tensor(tensor, projected_dim, projected_indices):
        dim_size = tensor.size()[projected_dim]
        for z in range(depth):
            for y in range(height):
                for x in range(width):
                    zero_entries = 0
                    value = 0
                    projected_index = -1
                    for w in range(dim_size):
                        i = [z, y, x]
                        i.insert(projected_dim, w)
                        element = tensor[torch.Size(i)].item()
                        if element == 0:
                            zero_entries += 1
                        else:
                            projected_index = w
                            value = element
                    
                    expected_projected_index = projected_indices[x + y * width + z * width * height].item()
                    assert projected_index == expected_projected_index
                    expected_value = 1 + x + y * width + z * width * height
                    assert zero_entries == (dim_size - 1)
                    assert value == expected_value


    # The value corresponds to the linear index + 1 to make it simple to test.
    values = torch.arange(0, element_count) + 1
    values = values.view(depth, height, width)

    # Test that expansion with dim_size 1 is equivalent to unsqueeze, such that project_tensor's dim is equivalent to torch's dim.
    zeros_indices = torch.zeros(element_count, dtype=torch.long)
    for dim in range(values.ndim+1):
        projected_values = project_tensor(values, dim, zeros_indices, dim_size=1)
        unsqueezed_values = values.unsqueeze(dim=dim)
        assert projected_values.shape == unsqueezed_values.shape
        equal_elements = torch.all(projected_values.eq(unsqueezed_values))
        assert equal_elements, "All elements in the two tensors should be equal."

    # Test expansion along different dimensions.
    projected_size = 5
    projected_indices = torch.arange(0, element_count, dtype=torch.long)
    projected_indices = torch.fmod(projected_indices, projected_size)
    for projected_dim in range(values.ndim+1):
        projected_values = project_tensor(values, projected_dim, projected_indices, projected_size)

        expected_shape = [depth, height, width]
        expected_shape.insert(projected_dim, projected_size)
        assert projected_values.shape == torch.Size(expected_shape)
        verify_tensor(projected_values, projected_dim, projected_indices)

if __name__ == "__main__":
    # test_project_tensor()

    width = 4
    height = 3
    depth = 3
    element_count = depth * height * width

    values = torch.arange(0, element_count) + 1
    values = values.view(depth, height, width)

    projected_size = 5
    projected_indices = torch.arange(0, element_count, dtype=torch.long)
    projected_indices = torch.fmod(projected_indices, projected_size)

    for projected_dim in range(values.ndim+1):
        projected_values = project_tensor(values, projected_dim, projected_indices, projected_size)
        unprojected_values = unproject_tensor(projected_values, projected_dim, projected_indices)
        
        assert values.shape == unprojected_values.shape
        equal_elements = torch.all(values.eq(unprojected_values))
        assert equal_elements, "All elements in the two tensors should be equal."