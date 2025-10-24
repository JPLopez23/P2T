import time
from typing import Dict, Set, List, Tuple, Optional
from collections import defaultdict
import copy
import os

# Clase que representa una gramática libre 
class Grammar:
    def __init__(self, name: str = ""):
        self.productions = defaultdict(list)
        self.terminals = set()
        self.non_terminals = set()
        self.start_symbol = 'S'
        self.name = name
        self.examples = {
            'correct': [],
            'syntactic_only': [],
            'incorrect': []
        }
    
    def add_production(self, lhs: str, rhs: List[str]):
        self.productions[lhs].append(rhs)
        self.non_terminals.add(lhs)

        for symbol in rhs:
            if symbol.islower() or symbol in ['a', 'the']:
                self.terminals.add(symbol)
            else:
                self.non_terminals.add(symbol)

    def add_example(self, category: str, sentence: str):
        if category in self.examples:
            self.examples[category].append(sentence)

    def print_grammar(self):
        for lhs in sorted(self.productions.keys()):
            for rhs in self.productions[lhs]:
                print(f"{lhs} -> {' '.join(rhs)}")

# Convierte una gramatica CFG a Forma Normal de Chomsky
class CNFConverter:
    def __init__(self, grammar: Grammar):
        self.grammar = copy.deepcopy(grammar)
        self.new_non_terminal_counter = 0
    
    def is_in_cnf(self) -> bool:
        for lhs, productions in self.grammar.productions.items():
            for rhs in productions:
                if len(rhs) > 2:
                    return False
                
                if len(rhs) == 1:
                    if rhs[0] not in self.grammar.terminals:
                        pass
                
                if len(rhs) == 2:
                    if rhs[0] in self.grammar.terminals or rhs[1] in self.grammar.terminals:
                        return False
        return True
    
    def convert_to_cnf(self) -> Grammar:
        if self.is_in_cnf():
            print("La gramática ya está en forma normal de Chomsky.")
            return self.grammar
        
        print("Convirtiendo la gramática a CNF...")
        self._replace_terminals_in_long_productions()
        self._break_long_productions()
        return self.grammar
    
    def _get_new_non_terminal(self) -> str:
        self.new_non_terminal_counter += 1
        return f"X{self.new_non_terminal_counter}"

    def _replace_terminals_in_long_productions(self):
        terminal_to_nt = {}
        new_productions = defaultdict(list)

        for lhs, productions in self.grammar.productions.items():
            for rhs in productions:
                if len(rhs) == 1:
                    new_productions[lhs].append(rhs)
                
                elif len(rhs) == 2:
                    new_rhs = []
                    for symbol in rhs:
                        if symbol in self.grammar.terminals:
                            if symbol not in terminal_to_nt:
                                new_nt = self._get_new_non_terminal()
                                terminal_to_nt[symbol] = new_nt
                                new_productions[new_nt].append([symbol])
                                self.grammar.non_terminals.add(new_nt)
                            new_rhs.append(terminal_to_nt[symbol])
                        else:
                            new_rhs.append(symbol)
                    new_productions[lhs].append(new_rhs)
                else:
                    new_productions[lhs].append(rhs)
        self.grammar.productions = new_productions
    
    def _break_long_productions(self):
        new_productions = defaultdict(list)

        for lhs, productions in self.grammar.productions.items():
            for rhs in productions:
                if len(rhs) <= 2:
                    new_productions[lhs].append(rhs)
                else:
                    current_lhs = lhs
                    for i in range(len(rhs) - 2):
                        new_nt = self._get_new_non_terminal()
                        new_productions[current_lhs].append([rhs[i], new_nt])
                        self.grammar.non_terminals.add(new_nt)
                        current_lhs = new_nt
                    
                    new_productions[current_lhs].append([rhs[-2], rhs[-1]])
        self.grammar.productions = new_productions

# Clase que representa un nodo en el árbol de parsing
class ParseTreeNode:
    def __init__(self, symbol: str, children: List['ParseTreeNode'] = None):
        self.symbol = symbol
        self.children = children or []

    def print_tree(self, level=0, prefix=""):
        indent = "  " * level
        print(f"{indent}{prefix}{self.symbol}")
        for i, child in enumerate(self.children):
            is_last = i == len(self.children) - 1
            child_prefix = "└─ " if is_last else "├─ "
            child.print_tree(level + 1, child_prefix)
    
    def to_string_tree(self, level=0) -> str:
        indent = "  " * level
        result = f"{indent}{self.symbol}\n"
        for child in self.children:
            result += child.to_string_tree(level + 1)
        return result

# Implementa el algoritmo CYK para el parsing de gramáticas en CNF
class CYKParser:
    def __init__(self, grammar: Grammar):
        self.grammar = grammar
        self.table = None
        self.back_pointer = None
    
    def parse(self, sentence: str) -> Tuple[bool, Optional[ParseTreeNode], float]:
        start_time = time.time()
        words = sentence.lower().split()
        n = len(words)

        if n == 0:
            return False, None, 0.0
        
        self.table = [[set() for _ in range(n)] for i in range(n)]

        self.back_pointer = [[defaultdict(list) for _ in range(n)] for _ in range(n)]

        for i in range(n):
            word = words[i]
            for lhs, productions in self.grammar.productions.items():
                for rhs in productions:
                    if len(rhs) == 1 and rhs[0] == word:
                        self.table[i][0].add(lhs)
                        self.back_pointer[i][0][lhs].append(('terminal', word, None))

        for length in range(2, n + 1):
            for i in range(n - length + 1):
                j = length - 1

                for k in range(length - 1):
                    left_symbols = self.table[i][k]
                    right_symbols = self.table[i + k + 1][j - k - 1]

                    for lhs, productions in self.grammar.productions.items():
                        for rhs in productions:
                            if len(rhs) == 2:
                                B, C = rhs
                                if B in left_symbols and C in right_symbols:
                                    self.table[i][j].add(lhs)
                                    self.back_pointer[i][j][lhs].append((k, B, C))
        
        accepted = self.grammar.start_symbol in self.table[0][n - 1]

        end_time = time.time()
        execution_time = end_time - start_time

        parse_tree = None
        if accepted:
            parse_tree = self._build_parse_tree(0, n - 1, self.grammar.start_symbol, words)
        return accepted, parse_tree, execution_time

    def _build_parse_tree(self, i: int, j: int, symbol: str, words: List[str]) -> ParseTreeNode:

        node = ParseTreeNode(symbol)

        if j == 0:
            pointers = self.back_pointer[i][j].get(symbol, [])
            if pointers:
                pointer = pointers[0]
                
                if pointer[0] == 'terminal':
                    terminal_node = ParseTreeNode(pointer[1])
                    node.children.append(terminal_node)
        else:
            pointers = self.back_pointer[i][j].get(symbol, [])
            if pointers:
                K, B, C = pointers[0]

                left_child = self._build_parse_tree(i, K, B, words)
                node.children.append(left_child)

                right_child = self._build_parse_tree(i + K + 1, j - K - 1, C, words)
                node.children.append(right_child)
        return node

    # Imprime la tabla CYK
    def print_table(self, words: List[str]):
        n = len(words)
        print("\nTabla CYK:")
        print("----------------------------------")
        for j in range(n - 1, -1, -1):
            for i in range(n - j):
                symbols = self.table[i][j]
                print(f"[{i},{j}]: {symbols if symbols else '∅'}", end="  ")
            print()
        print("----------------------------------")

# Función para cargar gramáticas desde archivo
def load_grammars_from_file(filename: str = "gramaticas.txt") -> List[Grammar]:
    grammars = []
    
    if not os.path.exists(filename):
        print(f"⚠ Advertencia: No se encontró el archivo '{filename}'")
        print("Creando gramática por defecto...")
        return [create_default_grammar()]
    
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
            
        grammar_blocks = content.split('[GRAMMAR]')
        
        for block in grammar_blocks:
            if not block.strip():
                continue
                
            lines = block.strip().split('\n')
            grammar = None
            current_section = None
            
            for line in lines:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                if line.startswith('NAME:'):
                    name = line.replace('NAME:', '').strip()
                    grammar = Grammar(name)
                
                elif line.startswith('[PRODUCTIONS]'):
                    current_section = 'productions'
                
                elif line.startswith('[EXAMPLES_CORRECT]'):
                    current_section = 'correct'
                
                elif line.startswith('[EXAMPLES_SYNTACTIC]'):
                    current_section = 'syntactic'
                
                elif line.startswith('[EXAMPLES_INCORRECT]'):
                    current_section = 'incorrect'
                
                elif grammar and current_section == 'productions':
                    if '->' in line:
                        parts = line.split('->')
                        lhs = parts[0].strip()
                        rhs_options = parts[1].split('|')
                        
                        for rhs in rhs_options:
                            symbols = rhs.strip().split()
                            grammar.add_production(lhs, symbols)
                
                elif grammar and current_section in ['correct', 'syntactic', 'incorrect']:
                    category_map = {
                        'correct': 'correct',
                        'syntactic': 'syntactic_only',
                        'incorrect': 'incorrect'
                    }
                    grammar.add_example(category_map[current_section], line)
            
            if grammar and grammar.productions:
                grammars.append(grammar)
        
        if not grammars:
            print("⚠ No se encontraron gramáticas válidas en el archivo.")
            print("Creando gramática por defecto...")
            return [create_default_grammar()]
            
    except Exception as e:
        print(f"⚠ Error al leer el archivo: {e}")
        print("Creando gramática por defecto...")
        return [create_default_grammar()]
    
    return grammars

# Función para crear la gramática por defecto
def create_default_grammar() -> Grammar:
    g = Grammar("English Grammar - Default")
    
    # S -> NP VP
    g.add_production('S', ['NP', 'VP'])
    
    # VP -> VP PP
    g.add_production('VP', ['VP', 'PP'])
    
    # VP -> V NP
    g.add_production('VP', ['V', 'NP'])
    
    # VP -> cooks | drinks | eats | cuts
    g.add_production('VP', ['cooks'])
    g.add_production('VP', ['drinks'])
    g.add_production('VP', ['eats'])
    g.add_production('VP', ['cuts'])
    
    # PP -> P NP
    g.add_production('PP', ['P', 'NP'])
        
    # NP -> Det N
    g.add_production('NP', ['Det', 'N'])
        
    # NP -> he | she
    g.add_production('NP', ['he'])
    g.add_production('NP', ['she'])
        
    # V -> cooks | drinks | eats | cuts
    g.add_production('V', ['cooks'])
    g.add_production('V', ['drinks'])
    g.add_production('V', ['eats'])
    g.add_production('V', ['cuts'])
        
    # P -> in | with
    g.add_production('P', ['in'])
    g.add_production('P', ['with'])
        
    # N -> cat | dog | beer | cake | juice | meat | soup | fork | knife | oven | spoon
    g.add_production('N', ['cat'])
    g.add_production('N', ['dog'])
    g.add_production('N', ['beer'])
    g.add_production('N', ['cake'])
    g.add_production('N', ['juice'])
    g.add_production('N', ['meat'])
    g.add_production('N', ['soup'])
    g.add_production('N', ['fork'])
    g.add_production('N', ['knife'])
    g.add_production('N', ['oven'])
    g.add_production('N', ['spoon'])
        
    # Det -> a | the
    g.add_production('Det', ['a'])
    g.add_production('Det', ['the'])
    
    # Ejemplos
    g.add_example('correct', 'she eats a cake')
    g.add_example('correct', 'the cat drinks the beer')
    g.add_example('syntactic_only', 'the fork eats a cat')
    g.add_example('syntactic_only', 'he drinks the oven')
    g.add_example('incorrect', 'she quickly eats cake')
    g.add_example('incorrect', 'cats drink beer')
        
    return g

# Función para seleccionar una gramática
def select_grammar(grammars: List[Grammar]) -> Optional[Grammar]:
    print("\n" + "="*70)
    print("GRAMÁTICAS DISPONIBLES")
    print("="*70)
    
    for i, grammar in enumerate(grammars, 1):
        print(f"{i}. {grammar.name}")
    
    print(f"{len(grammars) + 1}. Salir")
    
    while True:
        try:
            choice = input(f"\nSeleccione una gramática (1-{len(grammars) + 1}): ").strip()
            choice_num = int(choice)
            
            if choice_num == len(grammars) + 1:
                return None
            
            if 1 <= choice_num <= len(grammars):
                return grammars[choice_num - 1]
            else:
                print(f"✗ Por favor seleccione un número entre 1 y {len(grammars) + 1}")
        except ValueError:
            print("✗ Por favor ingrese un número válido")

# Función para ejecutar pruebas de ejemplo
def run_example_tests(parser: CYKParser, grammar: Grammar):
    print("\n" + "="*70)
    print("EJEMPLOS DE PRUEBA")
    print("="*70)
    
    if not any(grammar.examples.values()):
        print("\n⚠ No hay ejemplos definidos para esta gramática.")
        print("Puede usar el modo interactivo para probar frases.")
        return
    
    if grammar.examples['correct']:
        print("\n✓ EJEMPLOS SEMÁNTICAMENTE CORRECTOS:")
        print("-" * 70)
        
        for sentence in grammar.examples['correct']:
            accepted, parse_tree, exec_time = parser.parse(sentence)
            print(f"\nFrase: '{sentence}'")
            print(f"Resultado: {'SI' if accepted else 'NO'}")
            print(f"Tiempo: {exec_time:.6f} segundos")
            if accepted and parse_tree:
                print("Árbol de parsing:")
                parse_tree.print_tree()

    if grammar.examples['syntactic_only']:
        print("\n\n✓ EJEMPLOS SINTÁCTICAMENTE CORRECTOS Y SEMÁNTICAMENTE INCORRECTOS:")
        print("-" * 70)
        
        for sentence in grammar.examples['syntactic_only']:
            accepted, parse_tree, exec_time = parser.parse(sentence)
            print(f"\nFrase: '{sentence}'")
            print(f"Resultado: {'SI' if accepted else 'NO'}")
            print(f"Tiempo: {exec_time:.6f} segundos")
            if accepted and parse_tree:
                print("Árbol de parsing:")
                parse_tree.print_tree()
    
    if grammar.examples['incorrect']:
        print("\n\n✗ EJEMPLOS NO ACEPTADOS POR LA GRAMÁTICA:")
        print("-" * 70)
        
        for sentence in grammar.examples['incorrect']:
            accepted, parse_tree, exec_time = parser.parse(sentence)
            print(f"\nFrase: '{sentence}'")
            print(f"Resultado: {'SI' if accepted else 'NO'}")
            print(f"Tiempo: {exec_time:.6f} segundos")
            if not accepted:
                print(f"Razón: Contiene palabras o estructura no definida en la gramática")

# Función para mostrar vocabulario de la gramática
def show_vocabulary(grammar: Grammar):
    print("\nVocabulario disponible en la gramática:")
    print("="*70)
    
    # Agrupar terminales por categoría si es posible
    terminals_list = sorted(list(grammar.terminals))
    
    print(f"Terminales: {', '.join(terminals_list)}")
    print(f"\nNo terminales: {', '.join(sorted(list(grammar.non_terminals)))}")
    
    print("\nProducciones principales:")
    for lhs in sorted(grammar.productions.keys())[:5]:
        productions = grammar.productions[lhs]
        rhs_str = ' | '.join([' '.join(rhs) for rhs in productions[:3]])
        if len(productions) > 3:
            rhs_str += " ..."
        print(f"  {lhs} -> {rhs_str}")

# Función para el modo interactivo
def interactive_mode(parser: CYKParser, grammar: Grammar):
    print("\n" + "="*70)
    print("MODO INTERACTIVO - INGRESO DE FRASES")
    print("="*70)
    print(f"Gramática activa: {grammar.name}")
    print("\nInstrucciones: ")
    print("  • Ingrese frases usando el vocabulario de la gramática")
    print("  • Escriba 'salir' para terminar")
    print("  • Escriba 'ayuda' para ver el vocabulario disponible")
    if grammar.examples['correct']:
        print("  • Escriba 'ejemplos' para ver ejemplos válidos")

    while True:
        print("\n" + "-"*70)
        user_input = input("Ingrese una frase: ").strip()

        if user_input.lower() in ['salir', 'exit', 'quit', 'q']:
            print("Regresando al menú principal...")
            break

        if user_input.lower() in ['ayuda', 'help', 'h']:
            show_vocabulary(grammar)
            continue

        if user_input.lower() in ['ejemplos', 'examples', 'e']:
            if grammar.examples['correct']:
                print("\n✓ EJEMPLOS DE FRASES VÁLIDAS:")
                print("-"*70)
                for sentence in grammar.examples['correct']:
                    print(f"  • {sentence}")
            else:
                print("\n⚠ No hay ejemplos definidos para esta gramática.")
            continue

        if not user_input:
            print("Por favor ingrese una frase válida.")
            continue

        print(f"\nAnalizando frase: \"{user_input}\"")
        print("-"*70)

        accepted, parse_tree, exec_time = parser.parse(user_input)

        if accepted:
            print("✓ Resultado: SI")
            print(f"La frase pertenece al lenguaje generado por la gramática.")
            print(f"\nTiempo de ejecución: {exec_time:.6f} segundos")

            if parse_tree:
                print("\nÁRBOL DE PARSING:")
                print("-"*70)
                parse_tree.print_tree()
        else:
            print("✗ Resultado: NO")
            print(f"La frase NO pertenece al lenguaje generado por la gramática.")
            print(f"\nTiempo de ejecución: {exec_time:.6f} segundos")

def main():
    print("="*70)
    print("PROYECTO 2 - TEORÍA DE LA COMPUTACIÓN")
    print("="*70)
    
    # Cargar gramáticas desde archivo
    grammars = load_grammars_from_file()
    
    # Seleccionar gramática
    selected_grammar = select_grammar(grammars)
    
    if not selected_grammar:
        print("\n¡Gracias por usar el parser CYK!")
        return
    
    print("\n1. GRAMÁTICA ORIGINAL (CFG):")
    print("-"*70)
    print(f"Gramática: {selected_grammar.name}")
    print()
    selected_grammar.print_grammar()

    print("\n\n2. CONVERSIÓN A FORMA NORMAL DE CHOMSKY (CNF):")
    print("-"*70)
    converter = CNFConverter(selected_grammar)
    cnf_grammar = converter.convert_to_cnf()
    print("\nGramática en CNF:")
    cnf_grammar.print_grammar()

    parser = CYKParser(cnf_grammar)

    while True:
        print("\n\n" + "="*70)
        print("MENÚ PRINCIPAL")
        print("="*70)
        print(f"Gramática activa: {selected_grammar.name}")
        print()
        print("1. Ejecutar ejemplos de prueba")
        print("2. Modo interactivo - Ingresar frases manualmente")
        print("3. Cambiar de gramática")
        print("4. Salir")

        opcion = input("\nSeleccione una opción (1-4): ").strip()

        if opcion == '1':
            run_example_tests(parser, cnf_grammar)
        elif opcion == '2':
            interactive_mode(parser, cnf_grammar)
        elif opcion == '3':
            selected_grammar = select_grammar(grammars)
            if selected_grammar:
                print("\n1. GRAMÁTICA ORIGINAL (CFG):")
                print("-"*70)
                print(f"Gramática: {selected_grammar.name}")
                print()
                selected_grammar.print_grammar()

                print("\n\n2. CONVERSIÓN A FORMA NORMAL DE CHOMSKY (CNF):")
                print("-"*70)
                converter = CNFConverter(selected_grammar)
                cnf_grammar = converter.convert_to_cnf()
                print("\nGramática en CNF:")
                cnf_grammar.print_grammar()

                parser = CYKParser(cnf_grammar)
            else:
                print("\n¡Gracias por usar el parser CYK!")
                break
        elif opcion == '4':
            print("\n¡Gracias por usar el parser CYK!")
            print("="*70)
            break
        else:
            print("\n✗ Opción inválida. Por favor seleccione 1, 2, 3 o 4")

if __name__ == "__main__":
    main()