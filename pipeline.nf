// pipeline.nf
nextflow.enable.dsl=2

process embed {
    script:
    """
    python embed.py > embedded_data.txt
    """
}

process classify {
    input:
    file embedded_data_txt from embed.out

    script:
    """
    python classify.py $(cat embedded_data.txt)
    """
}
process helloWorld {
    input: 
      val cheears
    output:
      stdout
    """
    echo $cheers
    """
}

workflow {
    channel.of('ciao', 'hello', 'world!') | helloWorld | view 
}