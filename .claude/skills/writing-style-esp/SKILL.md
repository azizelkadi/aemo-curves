---
name: writing-style-esp
description: "Reglas para todo texto en español escrito en este proyecto para el usuario: comentarios de código, docstrings, ficheros markdown, READMEs, páginas de Confluence, tickets de Jira, descripciones de PR e informes. Español claro: palabras comunes, sin lenguaje figurado, sin prosa con estilo de IA, sin narrar la historia, títulos cortos. Úsala siempre que escribas o edites texto en español, y cuando el usuario invoque /writing-style-esp para revisar texto existente."
argument-hint: "[opcional: fichero, carpeta o página a revisar]"
---

# writing-style-esp

Estas reglas valen para todo texto en español escrito en este proyecto: comentarios de código, docstrings, ficheros *.md, READMEs, páginas de Confluence, tickets de Jira, descripciones de PR e informes.

El objetivo: quien lee el texto lo lee una vez, de arriba abajo, y entiende cada frase sin pararse. Apunta a un lector de 16 años: sencillo pero preciso, siempre con datos. Cada regla de abajo sirve a ese objetivo. Cuando dos reglas choquen, elige lo que sea más fácil de leer.

## Español claro

- *Elige siempre la palabra más común.* "usar", no "utilizar" ni "emplear". "empezar", no "iniciar". "también", no "asimismo" ni "adicionalmente". "sobre" o "unos", no "aproximadamente". "bastante", no "suficientemente". "hacer", no "efectuar". "cambiar", no "modificar". "pero", no "no obstante". "porque", no "debido a que". "para", no "con el fin de". Si una palabra más común dice lo mismo, la palabra difícil está mal.
- *Sin lenguaje figurado.* Nada de metáforas, refranes ni imágenes, incluidas las suaves que salen solas: "hoja de ruta", "a vista de pájaro", "poner sobre la mesa", "dar una vuelta de tuerca", "la punta del iceberg", "quedarse corto", "la letra pequeña", "sobre el papel", "arrojar luz", "de la mano de", "el meollo", "piedra angular". Prueba: si la frase describe un objeto o una acción física que no está de verdad ahí, cámbialo por el dato literal.
  - Mal: "Dos varas de medir enmarcan cada número."
  - Bien: "Cada modelo se compara con dos valores."
- *Los términos del proyecto no son lenguaje figurado.* Un término que el proyecto define (suelo, techo, ancla, pliegue, mundo de features) se queda, se escribe igual siempre, y se explica con palabras llanas la primera vez que aparece.
- *Sencillo nunca significa vago.* Mantén los nombres exactos, las rutas, los números y el comportamiento. La precisión viene de los datos, no de las palabras grandes.
- *Escribe frases completas y unidas.* Lo bastante largas para leer del tirón, unidas con "así que", "porque", "y", "pero". Ni fragmentos telegráficos ("Reintentos: 3. Si falla: error.") ni cadenas de 40 palabras. Una idea por frase, y parte una frase que pase de unas 30 palabras.
- *Di solo lo que el lector necesita.* Quita detalles enteros que no cambian lo que el lector sabe o hace. Acorta quitando contenido, nunca comprimiendo frases en fragmentos.
- *Una sola voz por documento.* Las páginas que explican y las propuestas usan "nosotros". Las especificaciones y los procedimientos son impersonales ("el modelo", "el trabajo") con imperativos secos para los pasos. No mezcles las dos en un mismo documento.
- *Usa -, nunca la raya larga —.*
- *Sin punto y coma en la prosa.* Parte en dos frases, o une con "así que", "y" o "pero". (El punto y coma dentro del código está bien.)
- *Para un inciso dentro de una frase, usa paréntesis.* No guiones a los lados ("- así -").

## Nada de texto con estilo de IA

El lector reconoce el texto generado por su forma. Quita la forma:

- *Abre con el dato llano, sin adorno.* Sin primeras frases dramáticas, sin poner escena, sin frases que suenen a cita.
- *Sin patrones retóricos hechos para el efecto.* Sin giros de dos frases ("La respuesta no es X. Es Y."), sin listas en negativo ("No es una herramienta. No es una librería. Es una plataforma."), sin preguntas retóricas, sin frase corta y dramática para rematar, sin ritmo de tres partes puesto para sonar bien ("rápido, sencillo y fiable"). Una lista de tres cosas reales está bien, un trío puesto por el sonido no.
- *Sin elogios ni drama sobre el propio trabajo.* Nada de "potente", "elegante", "robusto" (como elogio), "sin fisuras", "completo" o "exhaustivo" (como elogio), "profundizar", "clave" como adjetivo.
- *Sin relleno.* Nada de "Cabe destacar que", "Con el fin de", "Es importante señalar que", "Además" como comienzo de párrafo, "En esencia", "La realidad es que". Empieza con el dato.
- *Sin intensificadores vacíos.* "realmente", "en realidad", "simplemente", "verdaderamente", "muy". Un adverbio que lleva un dato ("unos 2M de filas", "a propósito sin ajustar") se queda.
- *Sin frases resumen que repitan lo que se acaba de decir.*
- *No cambies una palabra por elegancia.* El mismo objeto lleva el mismo nombre en cada frase. Un segmento sigue siendo "segmento".
- *Sin negrita para enfatizar dentro de la prosa, sin signos de exclamación, sin emoji.* La negrita marca un término definido la primera vez, una etiqueta, o la cabecera de una tabla.

La skill /stop-slop tiene el catálogo completo de patrones: frases de relleno prohibidas en references/phrases.md y estructuras prohibidas (contrastes binarios, listas en negativo, fragmentación dramática, falsa agencia) en references/structures.md. Ese catálogo está en inglés, pero las mismas estructuras están prohibidas en español. Úsalo como lista de comprobación. Tres de sus reglas se relajan aquí:

1. La voz pasiva y los sujetos impersonales son normales en documentación técnica ("el modelo se entrena con las otras siete ventanas", "la fila se marca").
2. Un contraste corto dentro de la frase que cierra una lectura errónea probable ("un objetivo, no un límite duro") está permitido. Solo se prohíbe el giro dramático de dos frases.
3. La voz de ensayo (un sujeto humano en cada frase) no se aplica a los documentos técnicos.

## Títulos

- Frases nominales cortas, de una a cuatro palabras: "Regla de evaluación", "Resumen", "Fichas de modelo", "Fuentes".
- *Nunca empieces un título con un artículo (El, La, Los, Las).* "Regla de evaluación", no "La regla de evaluación". La excepción rara es un título que queda mal sin el artículo.
- Sin preguntas, sin frases completas, sin ingenio. Solo la primera palabra en mayúscula.

## Hechos, no historia

El texto dice lo que es verdad ahora. Nunca cuenta la historia de cómo llegaron a ser el texto, el código o la página.

- Nada de "esto se cambió porque", "después de ver que", "antes X, ahora Y", "como se comentó", "en la última tarea".
- Nunca menciones la tarea, la sesión, la conversación ni la herramienta que produjo el texto. Al lector futuro no le sirve de nada saber que una página se escribió mientras se hacía otro trabajo.
- La historia de un cambio va en el mensaje del commit y en el PR, nunca en los ficheros ni en las páginas.
- No dejes constancia de quién tomó una decisión, cuándo, ni del proceso alrededor. Nada de "X decidió", nada de "esto anula el plan anterior", nada de marco de revisión o de comité. Escribe el hecho ya decidido.
- Nunca narres que algo se añadió, luego se quitó, y luego se volvió a añadir. Un documento escribe solo el diseño final, como si siempre hubiera sido así.
- *Describe lo que hay, no lo que falta.* No menciones las partes que el objeto no usa, y no expliques por qué se quitó, se excluyó o se retiró algo. Una cosa que no es parte del objeto no va en su texto. La razón de la exclusión vive en el registro del experimento, en el commit, o en el PR.
  - Mal: "El contador de reporteros nRepSeg se guarda pero no se ajusta: su valor identifica el mercado." (La columna no es una feature del modelo, así que la ficha del modelo no debe mencionarla.)
  - Bien: nada. Lista solo las features que el modelo usa.
  - Excepción: cuando un documento compara elementos por un campo fijo que todos contestan, "ninguno" es un dato, no ruido ("Features de sondeo: ninguna").
- Un hecho pasado puede ser justo lo que el lector necesita. Escríbelo como un hecho con fecha, no como una historia. "La config declara moe-v2, pero los números publicados se calcularon con moe-v1." No: "luego subimos la config a moe-v2".
- *Al mover un documento de un espacio de borrador (Confluence, un RFC) al repo o a un sitio, comprueba cada afirmación contra el código primero.* Deja solo lo que está implementado, con los números de la implementación, y borra las propuestas, las alternativas, los diseños retirados, los parámetros superados, las secciones de próximos pasos y el andamiaje de aprobación. Que una fuente esté aprobada no hace verdad hoy lo que se copia de ella: el código manda, el documento le sigue.

| Regla | Mal | Escribe en su lugar |
|---|---|---|
| No repitas lo que dice el código. | # sube el contador de arriba \n contador += 1 | Nada. Bórralo. |
| No cuentes lo que aprendiste al programar. | # Usa un set, tras depurar vimos que los duplicados rompían la ejecución. | # Los ids de entrada pueden repetirse, así que guarda solo el primero. |
| Nada de números de ticket en comentarios ni docs. | # Arreglo para OMA-4521. | Escribe la restricción real, o nada. El ticket va en el nombre de la rama y en el PR. |
| Nada de números ni nombres del fallo que acabas de arreglar. | # El timeout es 30s porque la ejecución de marzo tardó 25s. | # El servicio de arriba puede ser más lento que el timeout por defecto. |
| No adivines lo que querrán los que llamen al código en el futuro. | # Puede que quien llame quiera pasar su propio cliente más adelante. | Nada. Cíñete a los hechos del código tal como es. |

Excepción: los ficheros de plan bajo <área>/plans/ enlazan su ticket de Jira a propósito.

## Cuándo escribir un comentario o docstring

Por defecto, sin comentario. Si el código, el nombre de la función o la firma de tipos ya lo dicen, no lo repitas con palabras.

Antes de dejar un comentario, pregúntate: ¿confundiría al lector que conoce el lenguaje y ve la firma si lo quito? Si no, bórralo.

Un comentario se gana su sitio solo cuando dice algo que el lector no puede ver en el código:

- Una restricción oculta. # La API rechaza lotes de más de 500 filas.
- Un invariante no obvio. # Las filas llegan ordenadas por fecha, y el bucle depende de eso.
- Una razón real por la que la versión simple no funciona. # mypy estricto necesita el import del módulo que lo define

Los docstrings siguen la misma prueba: di algo que el nombre no diga ya. Sáltate """Calcula la puntuación.""" en def calcula_puntuacion. Las reglas de estructura (Args, Returns, Raises) están en CLAUDE.md.

## Revisar texto existente (/writing-style-esp [ruta])

Cuando el usuario invoca la skill sobre un fichero, una carpeta o una página:

1. Lee el texto en cuestión.
2. Marca cada incumplimiento con la regla que rompe.
3. Propón el arreglo. A menudo es borrar, y a veces una reescritura más corta y con datos.
4. Aplica los arreglos solo después de que el usuario lo confirme.
