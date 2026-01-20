# Cluster 6

def train(model, train_dataloader, val_dataloader, optimizer, scheduler, num_epochs, patience, clip_value):
    best_val_accuracy = 0.0
    epochs_without_improvement = 0
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_accuracy = 0
        for batch in tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            approaches = batch['approaches'].to(device)
            ranks = batch['ranks'].to(device)
            tokens = batch['tokens'].to(device)
            effort = (tokens - tokens.min()) / (tokens.max() - tokens.min())
            best_approach_indices = ranks.argmin(dim=1)
            logits = model(input_ids, attention_mask, effort[:, 0])
            ce_loss = F.cross_entropy(logits, best_approach_indices)
            effort_loss = F.mse_loss(logits.softmax(dim=1).gather(1, best_approach_indices.unsqueeze(1)).squeeze(), effort[:, 0])
            loss = ce_loss + 0.1 * effort_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=-1)
            total_accuracy += calculate_accuracy(predictions, best_approach_indices)
        avg_train_loss = total_loss / len(train_dataloader)
        avg_train_accuracy = total_accuracy / len(train_dataloader)
        avg_val_accuracy = validate(model, val_dataloader)
        print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Train Accuracy: {avg_train_accuracy:.4f}, Val Accuracy: {avg_val_accuracy:.4f}')
        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(avg_val_accuracy)
        else:
            scheduler.step()
        if avg_val_accuracy > best_val_accuracy:
            best_val_accuracy = avg_val_accuracy
            epochs_without_improvement = 0
            save_model(model, 'best_model.safetensors')
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            print(f'Early stopping triggered after {epoch + 1} epochs')
            break

def calculate_accuracy(predictions, labels):
    return (predictions == labels).float().mean()

def validate(model, val_dataloader):
    model.eval()
    total_val_accuracy = 0
    with torch.no_grad():
        for batch in val_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            approaches = batch['approaches'].to(device)
            ranks = batch['ranks'].to(device)
            tokens = batch['tokens'].to(device)
            effort = (tokens - tokens.min()) / (tokens.max() - tokens.min())
            best_approach_indices = ranks.argmin(dim=1)
            logits = model(input_ids, attention_mask, effort[:, 0])
            predictions = torch.argmax(logits, dim=-1)
            total_val_accuracy += calculate_accuracy(predictions, best_approach_indices)
    return total_val_accuracy / len(val_dataloader)

def main(args):
    if args.push_to_hub:
        base_model = AutoModel.from_pretrained(args.model_name)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        base_model.push_to_hub(args.hub_model_id)
        tokenizer.push_to_hub(args.hub_model_id)
        return
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    dataset = load_and_preprocess_data(tokenizer)
    kf = KFold(n_splits=args.k_folds, shuffle=True, random_state=42)
    best_val_accuracy = 0
    best_fold = 0
    for fold, (train_indices, val_indices) in enumerate(kf.split(dataset), 1):
        print(f'\nTraining Fold {fold}')
        train_sampler = SubsetRandomSampler(train_indices)
        val_sampler = SubsetRandomSampler(val_indices)
        train_dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=train_sampler)
        val_dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=val_sampler)
        base_model = AutoModel.from_pretrained(args.model_name)
        model = OptILMClassifier(base_model, num_labels=len(APPROACHES)).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=2, verbose=True)
        train(model, train_dataloader, val_dataloader, optimizer, scheduler, args.num_epochs, args.patience, args.clip_value)
        fold_val_accuracy = validate(model, val_dataloader)
        print(f'Fold {fold} Validation Accuracy: {fold_val_accuracy:.4f}')
        save_model(model, f'model_fold_{fold}.safetensors')
        if fold_val_accuracy > best_val_accuracy:
            best_val_accuracy = fold_val_accuracy
            best_fold = fold
            save_model(model, 'best_model.safetensors')
    print(f'\nBest performing model was from fold {best_fold} with validation accuracy {best_val_accuracy:.4f}')
    base_model = AutoModel.from_pretrained(args.model_name)
    best_model = OptILMClassifier(base_model, num_labels=len(APPROACHES))
    best_model.to(device)
    load_model(best_model, 'best_model.safetensors')
    best_model.eval()
    test_prompts = ['Maximize x + y subject to: x + 2y <= 10, x >= 0, y >= 0', 'Find the shortest path between nodes A and B in the given graph', 'Solve the Tower of Hanoi problem with 4 disks', 'Determine if the given number is prime', 'Find all possible combinations of coins that sum up to $1', 'Solve the equation: 2x^3 - 5x^2 + 3x - 7 = 0', 'Summarize the main points of the given article in three sentences', 'Describe the contents of the image, including any text present', "Find the Nash equilibrium for the prisoner's dilemma game", 'Solve the Sudoku puzzle given the following initial configuration', 'Find the optimal route for a salesperson visiting 10 cities', 'If all A are B, and some B are C, what can we conclude about A and C?', "Predict the stock price for the next week given the past year's data", 'Plan a path for a robot to navigate through a room with obstacles', 'Identify the sentiment and main topics in the following customer review', 'Prove that the square root of 2 is irrational', 'Design a policy for an agent to maximize its score in a given game environment', 'Find the most relevant documents in the corpus for the given query', 'Decrypt the following message encrypted with a simple substitution cipher', 'Simulate a quantum circuit with 3 qubits and measure the output', 'Generate a 3D model of a house based on the given floor plan', 'Find potential binding sites for a given protein sequence in a DNA strand', 'Given a set of logical statements, determine if the conclusion follows', 'Write a short story in the style of Edgar Allan Poe about a haunted lighthouse']
    effort_levels = [0.0, 0.2, 0.5, 0.8, 1.0]
    print('\nInference Examples:')
    for prompt in test_prompts:
        print(f'\nTest Prompt: {prompt}')
        results = inference(best_model, tokenizer, prompt, effort_levels)
        for effort, (approach, confidence) in zip(effort_levels, results):
            print(f'Effort: {effort:.1f}, Predicted Approach: {approach}, Confidence: {confidence:.4f}')

def load_and_preprocess_data(tokenizer):
    dataset = load_dataset('json', data_files='optillm_combined_dataset.jsonl')
    data_items = []
    for item in dataset['train']:
        prompt = item['prompt']
        results = item['results']
        if not results:
            continue
        valid_results = [result for result in results if result['rank'] is not None and 'tokens' in result]
        if len(valid_results) != 13:
            continue
        valid_results.sort(key=lambda x: APPROACHES.index(x['approach']))
        approaches = [result['approach'] for result in valid_results]
        ranks = [result['rank'] for result in valid_results]
        tokens = [result['tokens'] for result in valid_results]
        data_items.append({'prompt': prompt, 'approaches': approaches, 'ranks': ranks, 'tokens': tokens})
    print(f'Total data points: {len(data_items)}')
    print(f'Unique prompts: {len(set((item['prompt'] for item in data_items)))}')
    approach_counts = Counter((approach for item in data_items for approach in item['approaches']))
    print('Approach distribution:')
    for approach, count in approach_counts.items():
        print(f'  {approach}: {count}')
    return OptILMDataset([item['prompt'] for item in data_items], [item['approaches'] for item in data_items], [item['ranks'] for item in data_items], [item['tokens'] for item in data_items], tokenizer)

def inference(model, tokenizer, prompt, effort_levels):
    model.eval()
    with torch.no_grad():
        inputs = tokenizer(prompt, return_tensors='pt', max_length=MAX_LENGTH, truncation=True, padding='max_length')
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        results = []
        for effort in effort_levels:
            effort_tensor = torch.tensor([effort], dtype=torch.float).to(device)
            logits = model(input_ids, attention_mask, effort_tensor)
            probabilities = F.softmax(logits, dim=1)
            predicted_approach_index = torch.argmax(probabilities, dim=1).item()
            results.append((APPROACHES[predicted_approach_index], probabilities[0][predicted_approach_index].item()))
    return results

