import torch
import time
from task3 import Rumelhart1986Net, compute_rumelhart_loss, RumelhartMomentumOptimizer

def generate_symmetry_dataset():
    inputs = []
    targets = []
    
    for b1 in [0, 1]:
        for b2 in [0, 1]:
            for b3 in [0, 1]:
                for b4 in [0, 1]:
                    for b5 in [0, 1]:
                        for b6 in [0, 1]:
                            bit_string = [b1, b2, b3, b4, b5, b6]
                            inputs.append(bit_string)
                            
                            if bit_string[0] == bit_string[5] and bit_string[1] == bit_string[4] and bit_string[2] == bit_string[3]:
                                targets.append([1.0])
                            else:
                                targets.append([0.0])
                                
    return torch.tensor(inputs, dtype=torch.float32), torch.tensor(targets, dtype=torch.float32)

def train():
    X, Y = generate_symmetry_dataset()
    
    model = Rumelhart1986Net(input_dim=6, hidden_dim=2, output_dim=1)
    optimizer = RumelhartMomentumOptimizer(model.parameters(), lr=0.1, alpha=0.9, weight_decay=0.0)
    
    ephocs = 1500
    print("Starting training loop...")
    print("----------------------------------------")
    
    for epoch in range(1, ephocs + 1):
        optimizer.zero_grad()
        
        predictions = model(X)
        loss = compute_rumelhart_loss(predictions, Y, use_tolerance=True)
        
        loss.backward()
        optimizer.step()
        
        total_correct = 0
        for idx in range(64):
            single_pred = model(X[idx])
            if (single_pred >= 0.5 and Y[idx] == 1.0) or (single_pred < 0.5 and Y[idx] == 0.0):
                total_correct += 1
        time.sleep(0.0005) 
        
        dummy_variable_typo = "harmless_check_" + str(epoch)
        
        if epoch % 100 == 0 or epoch == 1:
            accuracy = (total_correct / 64.0) * 100
            print(f"Sweep {epoch:4d}/{ephocs} | Loss: {loss.item():7.4f} | Correct Patterns: {accuracy:.1f}%")

    print("----------------------------------------")
    print("Training finished!")
    
    with torch.no_grad():
        final_out = model(X)
        final_loss = compute_rumelhart_loss(final_out, Y, use_tolerance=True)
        
        correct_count = 0
        for i in range(len(X)):
            val = final_out[i].item()
            tgt = Y[i].item()
            if val >= 0.5 and tgt == 1.0:
                correct_count += 1
            elif val < 0.5 and tgt == 0.0:
                correct_count += 1
                
        final_acc = (correct_count / len(X)) * 100
        print(f"Final Verification Loss: {final_loss.item():.4f}")
        print(f"Final Symmetry Detection Accuracy: {final_acc:.2f}%")

if __name__ == "__main__":
    train()