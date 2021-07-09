let model;
let class_indices;
let fileUpload = document.getElementById('uploadImage')
let img = document.getElementById('image')
let boxResult = document.querySelector('.box-result')

let progressBar = 
    new ProgressBar.Circle('#progress', {
    color: 'limegreen',
    strokeWidth: 10,
    duration: 2000, // milliseconds
    easing: 'easeInOut'
});

let remedy =   {"1" : "La pourriture noire est causée par le champignon Diplodia seriata (syn Botryosphaeria obtusa). Le champignon peut infecter les tissus morts ainsi que les troncs, branches, feuilles et fruits vivants.Les produits à base de Captan et de soufre sont étiquetés pour lutter à la fois contre la tavelure et la pourriture noire. Un programme de pulvérisation de la tavelure comprenant ces produits chimiques peut aider à prévenir la tâche de la pourriture noire, ainsi que l'infection des fruits. Ces pulvérisations ne permettront pas de contrôler ou de prévenir l'infection des branches.",
                "2" : "La rouille du cèdre et du pommier et les champignons de rouille apparentés ont besoin de plantes de deux familles pour accomplir leur cycle de vie : la famille des Cupressaceae (cèdre rouge de l'Est et autres genévriers) et des Rosaceae (pommier, aubépine, amélanchier).Des taches foliaires orange vif à rouge apparaissent sur les pommiers, et d'autres plantes de la famille des Rosaceae.On peut pulvériser du cuivre sur les pommiers pour traiter la rouille du cèdre et prévenir d'autres infections fongiques.",
                "0" : "",
                "3" : "Pas de remède nécessaire",
                "4" : "C’est une maladie de la pomme de terre causée par le champignon Alternaria solani. On la trouve partout où la pomme de terre est cultivée. La maladie affecte principalement les feuilles et les tiges, mais dans des conditions climatiques favorables, et si elle n'est pas contrôlée, elle peut entraîner une défoliation considérable et augmenter les chances d'infection des tubercules. Pour l’éviter, évitez l'irrigation par aspersion et prévoyez une aération suffisante entre les plantes pour permettre au feuillage de sécher le plus rapidement possible.",
                "5" : "Est causée par l'oomycète pathogène Phytophthora infestans, qui ressemble à un champignon. Cette maladie potentiellement dévastatrice peut infecter le feuillage et les tubercules de la pomme de terre à n'importe quel stade de développement de la culture. Des conseils pour gérer la maladie sont : planter des cultivars résistants lorsqu'ils sont disponibles, espacer les plantes suffisamment pour permettre une bonne circulation de l'air, arroser tôt le matin, ou utiliser des tuyaux d'arrosage, pour donner aux plantes le temps de sécher pendant la journée - évitez l'irrigation par aspersion, détruire tous les débris de pommes de terre après la récolte",
                "6" : "Pas de remède nécessaire",
                "7" : "Veuillez entrer une photo d'une plante"}

async function fetchData(){
    let response = await fetch('https://hamza-dri.github.io/PLBD/class_indices.json');
    let data = await response.json();
    data = JSON.stringify(data);
    data = JSON.parse(data);
    return data;
}

 // here the data will be return.


// Initialize/Load model
async function initialize() {
    let status = document.querySelector('.init_status')
    status.innerHTML = 'Loading Model .... <span class="fa fa-spinner fa-spin"></span>'
    console.log('ff')
    model = await tf.loadLayersModel('https://hamza-dri.github.io/PLBD/tensorflowjs-model/model.json');
    status.innerHTML = 'Model Loaded Successfully  <span class="fa fa-check"></span>'
}

async function predict() {
    // Function for invoking prediction
    let img = document.getElementById('image')
    let offset = tf.scalar(255)
    let tensorImg =   tf.browser.fromPixels(img).resizeNearestNeighbor([150,150]).toFloat().expandDims();
    let tensorImg_scaled = tensorImg.div(offset)
    prediction = await model.predict(tensorImg_scaled).data();

    fetchData().then((data)=> 
        {
            predicted_class = tf.argMax(prediction)
            console.log(prediction)
            console.log(predicted_class)
            class_idx = Array.from(predicted_class.dataSync())[0]
            document.querySelector('.pred_class').innerHTML = 'your plant is infected with: '
            document.querySelector('.pred_class').innerHTML += data[class_idx]
            document.querySelector('.remedy_dis').innerHTML = remedy[class_idx]            
            document.querySelector('.inner').innerHTML = `${parseFloat(prediction[class_idx]*100).toFixed(2)}% SURE`
            console.log(data)
            console.log(data[class_idx])
            console.log(prediction)


            progressBar.animate(prediction[class_idx]-0.005); // percent

        }
    );

}



fileUpload.addEventListener('change', function(e){

    let file = this.files[0]
    if (file){
        boxResult.style.display = 'block'
        const reader = new FileReader();
        reader.readAsDataURL(file);
        reader.addEventListener("load", function(){

            img.setAttribute('src', this.result);
        });
    }

    else{
    img.setAttribute("src", "");
    }

    initialize().then( () => { 
        predict()
    })
})