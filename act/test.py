import numpy as np
import torch

def process_action_label(action, label, is_pad):
    horizon, dim = action.shape
    new_actions = torch.zeros_like(action)
    new_labels = torch.zeros_like(label)
    new_is_pad = torch.zeros_like(is_pad)

    current_action = action  # Shape: (horizon, dim)
    current_label = label  # Shape: (horizon,)
    current_is_pad = is_pad

    indices = []
    i = -1
    while i < horizon:
        if current_label[i] == 0 and i+2 < horizon:
            i += 2  # Skip next element
            indices.append(i)
        elif current_label[i] == 1:
            # Check the next 4 elements if they exist
            if i + 4 < horizon and torch.all(current_label[i:i + 4] == 1):
                i += 4  # Skip the next 3 elements
                indices.append(i)
            else:
                # Find the next 0 element if it exists
                next_zero = (current_label[i + 1:] == 0).nonzero(as_tuple=True)[0]
                if len(next_zero) > 0:
                    i = i + 1 + next_zero[0].item()
                    indices.append(i)
                else:
                    break  # No more 0s, stop
        else:
            i += 1
    
    # Use the indices to extract new action and label
    new_actions[:len(indices)] = current_action[indices]
    new_labels[:len(indices)] = current_label[indices]
    new_is_pad[:len(indices)] = current_is_pad[indices]

    return new_actions, new_is_pad

# Example usage
action = torch.randn(400, 3)  # Shape: (batch_size, horizon, dim)
is_pad = torch.randn(400, 1)
label = torch.randint(0, 2, (400,))  # Shape: (batch_size, horizon), values 0 or 1
print('label:',label)
new_action, new_label, new_is_pad = process_action_label(action, label, is_pad)
print("New Action Shape:", new_action.shape)
print("New Label:", new_label)
